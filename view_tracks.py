#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict
import random
import cv2
import numpy as np
from tqdm import tqdm, trange

from util.io import load_gz_json, decode_png
from util.box import Box


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('detection_file')
    parser.add_argument('video_dir')
    parser.add_argument('-o', '--out_file')

    # Additional options
    parser.add_argument('--court_dir')
    parser.add_argument('--mask_dir')
    excl_args = parser.add_mutually_exclusive_group()
    excl_args.add_argument('--feet_dir')

    parser.add_argument('--speedup', type=int, default=1)

    # Options for showing only detections
    parser.add_argument('-l', '--limit', type=int, default=25)
    parser.add_argument('-d', '--dilate', type=int, default=1,
                        help='Number of seconds before/after')
    parser.add_argument('--min_track_len', type=int, default=100)
    parser.add_argument('--reverse', action='store_true')

    # Options for showing entire video
    parser.add_argument('--full', action='store_true')
    return parser.parse_args()


def get_contiguous_dets(dets, min_len):
    intervals = []
    start = None
    prev_frame = -100
    for frame, frame_dets in dets:
        assert len(frame_dets) > 0
        if frame - 1 != prev_frame:
            if start is not None:
                if prev_frame - start > min_len:
                    intervals.append((start, prev_frame))
            start = frame
        prev_frame = frame

    if prev_frame - start > min_len:
        intervals.append((start, prev_frame))
    return intervals


PoseTrack_COCO_Keypoint_Ordering = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

LEFT_COLOR = (120, 200, 255)
RIGHT_COLOR = (120, 255, 200)
MID_COLOR = (200, 120, 255)
RACKET_COLOR = (0, 0, 0)

LEFT_BONES = [(5, 1), (7, 5), (9, 7), (11, 5), (13, 11), (15, 13)]
RIGHT_BONES = [(6, 1), (8, 6), (10, 8), (12, 6), (14, 12), (16, 14)]
MID_BONES = [(0, 2), (0, 1), (11, 12)]
UNKNOWN_BONES = [(3, 0), (4, 0)]    # Ear keypoints seem to be nonsense

# Additional bones in exported data
LEFT_FEET = [(15, 17), (15, 19)]
RIGHT_FEET = [(16, 18), (16, 20)]
RACKET = [(21, 22)]


def draw_box(frame, box_data, color):
    box = Box(*box_data['xywh'], score=box_data.get('score'))
    cv2.rectangle(
        frame, (int(box.x), int(box.y)),
        (int(box.x + box.w), int(box.y + box.h)),
        color, 1)

    if box.score:
        cv2.putText(frame, '{:0.3f}'.format(box.score),
                    (int(box.x), int(box.y) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)


MIN_JOINT_SCORE = 0.3


def draw_pose(frame, box_data):
    pose = np.mean(np.array(box_data['pose']), axis=0)
    for bones, color in (
        (MID_BONES, MID_COLOR), (LEFT_BONES, LEFT_COLOR),
        (RIGHT_BONES, RIGHT_COLOR), # (UNKNOWN_BONES, (0, 0, 0))
    ):
        for i, j in bones:
            if min(pose[i][2], pose[j][2]) < MIN_JOINT_SCORE:
                continue

            a = int(pose[i][0]), int(pose[i][1])
            b = int(pose[j][0]), int(pose[j][1])
            cv2.line(frame, a, b, color, 3)


TENNIS_COURT_LINES = [(0, 1), (1, 2), (2, 3), (3, 0),
                      (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]


def draw_tennis_court(frame, points):
    for i, j in TENNIS_COURT_LINES:
        a = int(points[i][0]), int(points[i][1])
        b = int(points[j][0]), int(points[j][1])
        cv2.line(frame, a, b, (0, 235, 255), 3)


MIN_FOOT_KP_CONF = 0.25
FOOT_BONES = [(0, 1), (0, 2), (0, 3)]
FOOT_KP_ORDERING = [
    'Ankle',
    'BigToe',
    'SmallToe',
    'Heel',
]


def draw_feet(frame, feet_data):
    for foot, color in [('lfoot', LEFT_COLOR), ('rfoot', RIGHT_COLOR)]:
        pose = feet_data[foot]
        for i, j in FOOT_BONES:
            if min(pose[i][2], pose[j][2]) < MIN_FOOT_KP_CONF:
                continue
            a = int(pose[i][0]), int(pose[i][1])
            b = int(pose[j][0]), int(pose[j][1])
            cv2.line(frame, a, b, color, 5)


def draw_train_pose(frame, keyp_arr):
    for bones, color in (
        (MID_BONES, MID_COLOR),
        (LEFT_BONES, LEFT_COLOR),
        (RIGHT_BONES, RIGHT_COLOR),
        # (UNKNOWN_BONES, (0, 255, 0))
        (LEFT_FEET, LEFT_COLOR),
        (RIGHT_FEET, RIGHT_COLOR),
        (RACKET, RACKET_COLOR)
    ):
        for i, j in bones:
            if max(i, j) >= keyp_arr.shape[0]:
                # Racket and feet may not be included
                continue
            a = int(keyp_arr[i, 0]), int(keyp_arr[i, 1])
            b = int(keyp_arr[j, 0]), int(keyp_arr[j, 1])
            cv2.line(frame, a, b, color, 3)


def draw_masks(frame, all_mask_data, alpha=0.33):
    mask_frame = np.zeros_like(frame)
    for mask_data in all_mask_data:
        x, y, w, h = [int(z) for z in mask_data['xywh']]
        channel = 2 if mask_data['class'] == 0 else 1
        mask = decode_png(mask_data['mask'])
        mask_frame[y:y + h, x:x + w, channel] = mask * 255
    blend = cv2.addWeighted(frame, 1. - alpha, mask_frame, alpha, 0.)

    # Update the affected pixels
    mask = np.max(mask_frame, axis=2) != 0
    frame[mask] = blend[mask]


def draw(frame, frame_num, person_dict, racket_dict, ball_dict, court_dict,
         feet_dict, mask_dict):
    cv2.putText(frame, str(frame_num), (10, 10 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 1)

    if court_dict is not None:
        court_data = court_dict.get(frame_num)
        if court_data is not None:
            draw_tennis_court(frame, court_data['points'])

    if mask_dict is not None:
        mask_data = mask_dict.get(frame_num)
        if mask_data is not None:
            draw_masks(frame, mask_data)

    for box_data in person_dict.get(frame_num, []):
        draw_box(frame, box_data, (0, 0, 255))
        if 'pose' in box_data:
            draw_pose(frame, box_data)

    for box_data in racket_dict.get(frame_num, []):
        draw_box(frame, box_data, (255, 255, 255))

    for box_data in ball_dict.get(frame_num, []):
        draw_box(frame, box_data, (255, 255, 0))

    if feet_dict is not None:
        for feet_data in feet_dict.get(frame_num, []):
            draw_feet(frame, feet_data)


def get_video_name(fname):
    return fname.rsplit('/', 1)[-1].split('.', 1)[0]


def visualize_one(detection_file, args):
    all_dets = load_gz_json(detection_file)

    video_name = get_video_name(detection_file)
    video_path = os.path.join(args.video_dir, video_name + '.mp4')
    assert os.path.isfile(video_path)

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames < args.min_track_len:
        args.full = True
    gap = max(int(1000 / fps / args.speedup), 1)

    vo = None
    if args.out_file is not None:
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        vo = cv2.VideoWriter(
            args.out_file, cv2.VideoWriter_fourcc(*'h264'),
            fps, (width, height))

    person_dict = dict(all_dets['person'])
    racket_dict = dict(all_dets.get('racket', []))
    ball_dict = dict(all_dets.get('ball', []))

    court_dict = None
    if args.court_dir is not None:
        court_path = os.path.join(
            args.court_dir, video_name + '.court.json.gz')
        court_dict = dict(load_gz_json(court_path))

    feet_dict = None
    if args.feet_dir is not None:
        feet_path = os.path.join(
            args.feet_dir, video_name + '.feet.json.gz')
        feet_dict = dict(load_gz_json(feet_path))

    mask_dict = None
    if args.mask_dir is not None:
        mask_path = os.path.join(args.mask_dir, video_name + '.mask.json.gz')
        if os.path.exists(mask_path):
            mask_dict = dict(load_gz_json(mask_path))

    def show(frame, pause):
        if vo is None:
            cv2.imshow('frame', frame)
            cv2.waitKey(pause)
        else:
            vo.write(frame)

    if not args.full:
        # Show detections only
        contiguous_person_dets = get_contiguous_dets(
            all_dets['person'], args.min_track_len)
        frames_in_dets = sum(b - a for a, b in contiguous_person_dets)
        print('Found {} contiguous detections, totaling {} / {} frames and {:d} / {:d} seconds'.format(
            len(contiguous_person_dets), frames_in_dets, num_frames,
            int(frames_in_dets / fps), int(num_frames / fps)))

        if args.limit > 0:
            print('Showing {} results (see --limit)'.format(args.limit))
            random.shuffle(contiguous_person_dets)
        else:
            args.limit = 0xFFFFFFFF

        dilate_num_frames = int(fps * args.dilate)
        for a, b in tqdm(sorted(contiguous_person_dets[:args.limit],
                                reverse=args.reverse)):
            a -= dilate_num_frames
            b += dilate_num_frames
            pad_front = dilate_num_frames
            if a < 0:
                pad_front += a
            a = max(a, 0)

            vc.set(cv2.CAP_PROP_POS_FRAMES, a)
            for frame_num in range(a, b):
                ret, frame = vc.read()
                if not ret:
                    break

                # Fade in the padding
                if frame_num - a < pad_front:
                    frame = (frame.astype(np.float64) *
                            (frame_num - a) / dilate_num_frames
                    ).astype(np.uint8)
                if b - frame_num < dilate_num_frames:
                    frame = (frame.astype(np.float64) *
                            (b - frame_num) / dilate_num_frames
                    ).astype(np.uint8)

                draw(frame, frame_num, person_dict, racket_dict, ball_dict,
                     court_dict, feet_dict, mask_dict)
                show(frame, gap)

    else:
        # Show all frames only
        for frame_num in trange(num_frames):
            ret, frame = vc.read()
            if not ret:
                break
            draw(frame, frame_num, person_dict, racket_dict, ball_dict,
                 court_dict, feet_dict, mask_dict)
            show(frame, gap)

    vc.release()

    if vo is not None:
        vo.release()


def main(args):
    if os.path.isdir(args.detection_file):
        assert args.out_file is None
        detection_dir = args.detection_file
        for file_name in sorted(os.listdir(detection_dir)):
            visualize_one(os.path.join(detection_dir, file_name), args)
    else:
        visualize_one(args.detection_file, args)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_args())