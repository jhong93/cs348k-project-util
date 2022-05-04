#!/usr/bin/env python3
"""
This needs to copied and run in the DCPose/demo directory.
NOTE: Also need to download their model files.
"""

import os
import sys
import argparse
import json
import gzip
import logging
import numpy as np
import cv2
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath('../')

sys.path.insert(0, ROOT_DIR)

from posetimation.zoo import build_model
from posetimation.config import get_cfg, update_config
from datasets.process import get_final_preds
from datasets.transforms import build_transforms
from datasets.process import get_affine_transform
from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from engine.core.vis_helper import \
    add_poseTrack_joint_connection_to_image, add_bbox_in_image
from utils.utils_bbox import box2cs
from utils.common import INFERENCE_PHASE


def default_internal_args(gpu_id=0):
    parser = argparse.ArgumentParser(description='Inference pose estimation Network')

    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str,
                        default="./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml")
    parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
                        default='DcPose')
    parser.add_argument('-weight', help='model weight file', required=False, type=str
                        , default='./DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # philly
    args = parser.parse_args('')
    args.gpu_id = str(gpu_id)
    args.rootDir = ROOT_DIR
    args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
    args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
    return args


def get_inference_model(gpu_id):
    logger = logging.getLogger(__name__)
    global cfg, args
    args = default_internal_args(gpu_id)
    cfg = get_cfg(args)
    update_config(cfg, args)
    logger.info("load :{}".format(args.weight))
    checkpoint_dict = torch.load(args.weight)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    new_model = build_model(cfg, INFERENCE_PHASE)
    new_model.load_state_dict(model_state_dict)
    return new_model


MODEL = None
IMAGE_TRANSFORMS = build_transforms(None, INFERENCE_PHASE)
IMAGE_SIZE = np.array([288, 384])
ASPECT_RATIO = IMAGE_SIZE[0] * 1.0 / IMAGE_SIZE[1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir')
    parser.add_argument('track_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--part', type=int, nargs=2)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp)


def preprocess_frame(frame, prev_frame, next_frame, center, scale):
    trans_matrix = get_affine_transform(center, scale, 0, IMAGE_SIZE)
    frame = cv2.warpAffine(
        frame, trans_matrix, (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    frame = IMAGE_TRANSFORMS(frame)
    if prev_frame is None:
        prev_frame = frame
    else:
        prev_frame = cv2.warpAffine(
            prev_frame, trans_matrix, (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        prev_frame = IMAGE_TRANSFORMS(prev_frame)

    if next_frame is None:
        next_frame = frame
    else:
        next_frame = cv2.warpAffine(
            next_frame, trans_matrix, (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        next_frame = IMAGE_TRANSFORMS(next_frame)
    return frame, prev_frame, next_frame


FLIP_IDXS = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def process_frame(frame, prev_frame, next_frame, box_xywh):
    center, scale = box2cs(box_xywh, ASPECT_RATIO)
    target_image_data, prev_image_data, next_image_data = preprocess_frame(
        frame, prev_frame, next_frame, center, scale)

    target_image_data = target_image_data.unsqueeze(0)
    prev_image_data = prev_image_data.unsqueeze(0)
    next_image_data = next_image_data.unsqueeze(0)

    concat_input = torch.cat(
        (target_image_data, prev_image_data, next_image_data), 1).cuda()

    # Add flip to batch
    concat_input = torch.cat(
        (concat_input, torch.flip(concat_input, (3,))), dim=0)

    # margin = torch.stack(
    #     [torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1
    # ).cuda()
    margin = torch.ones((2, 2), dtype=torch.long, device=concat_input.device)

    predictions = MODEL(concat_input, margin=margin)

    # Unflip predictions for 2nd
    predictions[1] = predictions[1, FLIP_IDXS, :, :]
    predictions[1] = torch.flip(predictions[1], (2,))

    pred_joint, pred_conf = get_final_preds(
        predictions.cpu().detach().numpy(), [center, center], [scale, scale])
    pred_keypoints = np.concatenate(
        [pred_joint.astype(int), pred_conf], axis=2)
    return pred_keypoints


def vis_frame(frame, pred, bbox, window_name='frame'):
    new_coord = coco2posetrack_ord_infer(pred)
    img = add_poseTrack_joint_connection_to_image(
         frame, new_coord, sure_threshold=0.3, flag_only_draw_sure=True)
    xyxy_box = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    img = add_bbox_in_image(img, xyxy_box)
    cv2.imshow(window_name, img)
    cv2.waitKey(100)


def process_multi_track_video(
        video_file, track_path, pose_path, video_range=None
):
    vc = cv2.VideoCapture(video_file)
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    print('Reading {} ({}x{} @ {} fps)'.format(video_file, width, height, fps))
    if video_range:
        vc.set(cv2.CAP_PROP_POS_FRAMES, video_range[0])
        frame_count = video_range[1] - video_range[0] + 1
    else:
        frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    def postprocess(box_xywh, kp):
        if pose_path is None:
            vis_frame(frame.copy(), kp[0, :, :], box_xywh)
            vis_frame(frame.copy(), kp[1, :, :], box_xywh, window_name='hflip')
        return kp.tolist()

    # Directly add results to track_data
    track_data = load_gz_json(track_path)
    person_dict = dict(track_data['person'])

    with tqdm(total=len(person_dict)) as pbar:
        buffer = [None, None]
        for i in range(frame_count):
            ret, next_frame = vc.read()
            if not ret:
                break

            frame_bboxes = person_dict.get(i - 1)
            if frame_bboxes is not None:
                for bbox_data in frame_bboxes:
                    box_xywh = bbox_data['xywh']
                    prev_frame, frame = buffer
                    if frame is not None:
                        pred_kp = process_frame(
                            frame, prev_frame, next_frame, box_xywh)
                        bbox_data['pose'] = postprocess(box_xywh, pred_kp)
                pbar.update(1)

            buffer.append(next_frame)
            if len(buffer) > 2:
                buffer.pop(0)

        frame_bboxes = person_dict.get(i)
        if frame_bboxes is not None:
            for bbox_data in frame_bboxes:
                box_xywh = bbox_data['xywh']
                prev_frame, frame = buffer
                pred_kp = process_frame(frame, prev_frame, None, box_xywh)
                bbox_data['pose'] = postprocess(box_xywh, pred_kp)
            pbar.update(1)

    vc.release()
    if pose_path is None:
        cv2.destroyAllWindows()
    else:
        store_gz_json(pose_path, track_data)


def init_model(gpu):
    global MODEL
    MODEL = get_inference_model(gpu)
    MODEL = MODEL.cuda()
    MODEL.eval()


def main(video_dir, track_dir, out_dir, part, gpu, debug):
    init_model(gpu)

    video_files = sorted(os.listdir(video_dir))
    if part is not None:
        a, b = part
        video_files = [v for i, v in enumerate(video_files) if i % b == a]

    os.makedirs(out_dir, exist_ok=True)
    for video_file in tqdm(video_files, desc='Videos'):
        video_name = os.path.splitext(video_file)[0]
        if debug:
            pose_path = None
        else:
            pose_path = os.path.join(out_dir, video_name + '.pose.json.gz')
            if os.path.exists(pose_path):
                print('Already done:', pose_path)
                continue
        video_path = os.path.join(video_dir, video_file)
        track_path = os.path.join(track_dir, video_name + '.track.json.gz')
        if not os.path.exists(track_path):
            print('Missing tracking:', video_name)
            continue
        process_multi_track_video(video_path, track_path, pose_path)


if __name__ == '__main__':
    main(**vars(get_args()))
