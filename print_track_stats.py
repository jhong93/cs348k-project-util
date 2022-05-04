#!/usr/bin/env python3

import os
import argparse
from tabulate import tabulate
from tqdm import tqdm

from util.video import get_metadata
from util.io import load_gz_json
from view_tracks import get_contiguous_dets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('track_dir')
    parser.add_argument('video_dir')
    parser.add_argument('-l', '--min_length', type=int, default=100)
    return parser.parse_args()


def main(track_dir, video_dir, min_length):
    video_dict = {}
    for video_file in tqdm(
            os.listdir(video_dir), desc='Listing videos'
    ):
        if not video_file.endswith('.mp4'):
            continue
        video_name = os.path.splitext(video_file)[0]
        video_dict[video_name] = get_metadata(os.path.join(
            video_dir, video_file))

    rows = []
    total_seconds = 0.
    total_intervals = 0
    total_frames = 0
    total_poses = 0
    for track_file in tqdm(
            sorted(os.listdir(track_dir)), desc='Parsing tracks'
    ):
        track_path = os.path.join(track_dir, track_file)
        tracks = load_gz_json(track_path)
        intervals = get_contiguous_dets(tracks['person'], min_length)
        poses = sum(len(b) for _, b in tracks['person'])
        frames = sum(b - a + 1 for a, b in intervals)

        video_name = track_file.split('.', 1)[0]
        video_meta = video_dict[video_name]

        rows.append((
            video_name, len(intervals), frames, round(frames / video_meta.fps),
            poses, round(frames / video_meta.num_frames * 100, 2)
        ))

        total_frames += frames
        total_seconds += frames / video_meta.fps
        total_poses += poses
        total_intervals += len(intervals)

    print(tabulate(rows, headers=[
        'name', '# intervals', '# frames', '# seconds', '# poses', '%']))

    print('\nTotal intervals:', total_intervals)
    print('Total frames:', total_frames)
    print('Total seconds:', round(total_seconds))
    print('Total poses:', total_poses)


if __name__ == '__main__':
    main(**vars(get_args()))