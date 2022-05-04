import os
import cv2
from collections import namedtuple


VideoMetadata = namedtuple('VideoMetadata', [
    'fps', 'num_frames', 'width', 'height'
])


def _get_metadata(vc):
    fps = vc.get(cv2.CAP_PROP_FPS)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMetadata(fps, num_frames, width, height)


def get_metadata(video_path):
    vc = cv2.VideoCapture(video_path)
    try:
        return _get_metadata(vc)
    finally:
        vc.release()


def get_frame_num(file_name):
    return int(os.path.splitext(file_name)[0])


def save_frames_to_video(video_out_file, video_frame_dir):
    frame_files = [(get_frame_num(x), x) for x in os.listdir(video_frame_dir)]
    frame_files.sort(key=lambda x: x[0])

    vo = None
    for i, frame_file in frame_files:
        frame = cv2.imread(os.path.join(video_frame_dir, frame_file))
        if vo is None:
            height, width, _ = frame.shape
            vo = cv2.VideoWriter(
                video_out_file, cv2.VideoWriter_fourcc(*'avc1'),
                25, (width, height))    # NOTE: fps is arbitrary here
        vo.write(frame)
    vo.release()