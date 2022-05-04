#!/usr/bin/env python3
"""
Example code for heuristically extracting the players given detections
"""

import os
import argparse
from collections import defaultdict, Counter
from multiprocessing import Pool
import cv2
import numpy as np
from tqdm import tqdm

from util.io import load_gz_json, load_text, store_gz_json
from util.box import Box
from util.track import find_hard_cuts
from util.video import get_metadata, VideoMetadata
from util.sort import Sort


COCO_NAMES = load_text('util/data/coco-names.txt')
COCO_NAMES_INV = {v: i for i, v in enumerate(COCO_NAMES)}

COCO_FILTER_SET = {
    COCO_NAMES_INV['person'],
    COCO_NAMES_INV['sports ball'],
    COCO_NAMES_INV['tennis racket']
}

CURR_VIDEO_FILE = None

DUMMY_VIDEO_META = VideoMetadata(None, None, 1920, 1080)

MAX_PLAYER_HEIGHT_BY_TOURNAMENT = {
    'wimbledon': 0.4,
    'usopen': 0.3,
    'ausopen': 0.3
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_det_file')
    parser.add_argument('frame_diff_file')
    parser.add_argument('--video_file')
    parser.add_argument('-o', '--out_file')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-j', '--parallelism', type=int,
                        default=os.cpu_count() // 4)
    return parser.parse_args()


def display_boxes(video_path, tracked_boxes_dict, other_boxes_dict=None,
                  speed=1, out_file=None, frame_num=0):
    OTHER_BOX_COLOR = (255, 255, 255)
    TRACKED_BOX_COLOR = (0, 0, 255)

    vc = cv2.VideoCapture(video_path)
    if frame_num != 0:
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    fps = vc.get(cv2.CAP_PROP_FPS)
    gap = int(1000 / fps / speed)

    vo = None
    if out_file is not None:
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        vo = cv2.VideoWriter(
            out_file, cv2.VideoWriter_fourcc(*'mp4h'),
            fps, (width, height))

    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if other_boxes_dict is not None:
            for box in other_boxes_dict.get(frame_num, []):
                cv2.rectangle(
                    frame, (int(box.x), int(box.y)),
                    (int(box.x + box.w), int(box.y + box.h)),
                    OTHER_BOX_COLOR, 1)
                cv2.putText(frame, '{:0.3f}'.format(box.score),
                            (int(box.x), int(box.y) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, OTHER_BOX_COLOR, 1)

        if tracked_boxes_dict is not None:
            for box in tracked_boxes_dict.get(frame_num, []):
                cv2.rectangle(
                    frame, (int(box.x), int(box.y)),
                    (int(box.x + box.w), int(box.y + box.h)),
                    TRACKED_BOX_COLOR, 1)
                cv2.putText(frame, '{:0.3f}'.format(box.score),
                            (int(box.x), int(box.y) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, TRACKED_BOX_COLOR, 1)

        if vo is None:
            cv2.imshow('frame', frame)
            cv2.waitKey(gap)
        else:
            vo.write(frame)
        frame_num += 1
    vc.release()
    if vo is None:
        cv2.destroyWindow('frame')
    else:
        vo.release()


def between(x, a, b):
    assert a <= b, (a, b)
    return x >= a and x <= b


def find_potential_anchors(tournament, obj_dets, video_meta):
    racket_id = COCO_NAMES_INV['tennis racket']
    is_wimbledon = tournament == 'wimbledon'

    MAX_HEIGHT_IN_FRAME_CONTAINING_ANCHOR_PX = \
        MAX_PLAYER_HEIGHT_BY_TOURNAMENT[tournament] * video_meta.height

    TOP_THIRD_PX = video_meta.height * 0.3

    potential_anchors = {}
    for frame, dets in obj_dets[COCO_NAMES_INV['person']].items():
        exclude = False
        if len(dets) < 3:
            exclude = True
        elif frame not in obj_dets[racket_id]:
            exclude = True
        else:
            for b in dets:
                # Eliminate implausible close up cameras
                if b.h > MAX_HEIGHT_IN_FRAME_CONTAINING_ANCHOR_PX:
                    exclude = True
                    break

        if not exclude:
            rackets = obj_dets[racket_id][frame]

            anchors = []
            for d in dets:
                if d.score < 0.5:
                    continue
                if (d.h < d.w * 1.5
                    and not between(d.cx / video_meta.width, 0.3, 0.7)
                ):
                    # Not standing and off to the side
                    continue
                if d.y < TOP_THIRD_PX and is_wimbledon:
                    # Try to separate out refs in the background area
                    is_possible_ref = False
                    for d2 in dets:
                        if (d2._id != d._id
                            and d2.y < TOP_THIRD_PX
                            and d2.score > 0.5
                            and between(d2.cx / video_meta.width, 0.2, 0.8)
                        ):
                            # Must be the lowest among those in the top band
                            if d.y < d2.y:
                                is_possible_ref = True
                    if is_possible_ref:
                        continue

                for r in rackets:
                    # Should overlap a racket
                    if r.score > 0.5 and d.iou(r) > 0:
                        anchors.append(d)

            if len(anchors) > 0 and len(anchors) <= 3:
                # Don't want too many anchors too in one frame
                potential_anchors[frame] = anchors

    return potential_anchors


def track_bboxes_naive(
        tournament, anchor_dets, person_dets, video_meta, min_iou=0.5,
        enforce_size_limits=True, boundaries=None
):
    MAX_PLAYER_HEIGHT_PX = \
        MAX_PLAYER_HEIGHT_BY_TOURNAMENT[tournament] * video_meta.height

    # Forward tracking
    fwd_track_dets = defaultdict(list)
    for frame in sorted(anchor_dets):
        for anchor in anchor_dets[frame]:
            prev = anchor
            i = frame + 1
            while True:
                done = False

                # Have we hit another anchor?
                if i in anchor_dets:
                    for a in anchor_dets[i]:
                        if prev.iou(a) > min_iou:
                            done = True
                            break
                if done:
                    break

                if enforce_size_limits:
                    # Do not track into frames that have likely camera changes
                    for p in person_dets[i]:
                        if p.h > MAX_PLAYER_HEIGHT_PX and p.score > 0.8:
                            done = True
                            break
                    if done:
                        break

                # Have we hit a boundary?
                if boundaries is not None:
                    if i in boundaries:
                        # print('    hit boundary')
                        break

                # Track among non-anchors
                best = None
                best_iou = min_iou
                for p in person_dets[i]:
                    iou = prev.iou(p)
                    if iou > best_iou:
                        best = p
                        best_iou = iou
                if best is not None:
                    fwd_track_dets[i].append(best)
                    prev = best
                else:
                    # Lost track
                    break
                i += 1

    bck_track_dets = defaultdict(list)
    for frame in sorted(anchor_dets):
        for anchor in anchor_dets[frame]:
            prev = anchor
            i = frame - 1
            while True:
                done = False

                # Have we hit another anchor?
                if i in anchor_dets:
                    for a in anchor_dets[i]:
                        if prev.iou(a) > min_iou:
                            done = True
                            break
                if done:
                    break

                if enforce_size_limits:
                    # Do not track into frames that have likely camera changes
                    for p in person_dets[i]:
                        if p.h > MAX_PLAYER_HEIGHT_PX and p.score > 0.8:
                            done = True
                            break
                    if done:
                        break

                # Have we hit a boundary?
                if boundaries is not None:
                    if i + 1 in boundaries:
                        # print('    hit boundary')
                        break

                # Track among non-anchors
                best = None
                best_iou = min_iou
                for p in person_dets[i]:
                    iou = prev.iou(p)
                    if iou > best_iou:
                        best = p
                        best_iou = iou
                if best is not None:
                    bck_track_dets[i].append(best)
                    prev = best
                else:
                    # Lost track
                    break
                i -= 1

    for frame in anchor_dets:
        for b in anchor_dets[frame]:
            fwd_track_dets[frame].append(b)
    for frame in bck_track_dets:
        for b in bck_track_dets[frame]:
            fwd_track_dets[frame].append(b)

    for frame in fwd_track_dets:
        seen = set()
        dedup = []
        for b in fwd_track_dets[frame]:
            if b._id not in seen:
                seen.add(b._id)
                dedup.append(b)
        fwd_track_dets[frame] = sorted(dedup, key=lambda x: x.score)
    return fwd_track_dets


def compute_sort_tracks(anchor_dets, max_age=3):
    prev_frame = None
    tracker = Sort(max_age=max_age)
    tracks = []
    for frame in sorted(anchor_dets):
        dets = anchor_dets[frame]

        # Advance state
        if prev_frame is not None and prev_frame + 1 < frame:
            while prev_frame + 1 < frame:
                tracker.update(np.empty((0, 5)))
                prev_frame += 1

        boxes = []
        for b in dets:
            boxes.append([b.x, b.y, b.x2, b.y2, b.score])

        frame_tracks = tracker.update(np.array(boxes))
        if frame_tracks.shape[0] > 0:
            tmp = []
            for i in range(frame_tracks.shape[0]):
                x1, y1, x2, y2, obj_id = frame_tracks[i, :].tolist()
                b = Box(x1, y1, x2 - x1, y2 - y1)

                # Match with detections
                best = None
                best_iou = 0.5
                for d in dets:
                    iou = d.iou(b)
                    if iou > best_iou:
                        best = d
                        best_iou = iou

                if best is not None:
                    tmp.append((int(obj_id), best))
            tracks.append((frame, tmp))

        prev_frame = frame
    return tracks


def filter_implausible_balls_with_sort(ball_dets, video_meta):
    sort_tracks = compute_sort_tracks(ball_dets, max_age=5)

    track_count = Counter()
    track_min_cy = defaultdict(lambda: float('inf'))
    track_max_cy = defaultdict(lambda: float('-inf'))
    track_min_cx = defaultdict(lambda: float('inf'))
    track_max_cx = defaultdict(lambda: float('-inf'))
    tracked_box_ids = set()
    for frame, track_and_box in sort_tracks:
        for track_id, box in track_and_box:
            track_count[track_id] += 1
            track_min_cy[track_id] = min(track_min_cy[track_id], box.cy)
            track_max_cy[track_id] = max(track_max_cy[track_id], box.cy)
            track_min_cx[track_id] = min(track_min_cx[track_id], box.cx)
            track_max_cx[track_id] = max(track_max_cx[track_id], box.cx)
            tracked_box_ids.add(box._id)

    # Delete static tracks
    track_del_set = set()
    for track_id in track_min_cy:
        if track_count[track_id] > 5 and max(
            (track_max_cy[track_id] - track_min_cy[track_id]) /
                video_meta.height,
            (track_max_cx[track_id] - track_min_cx[track_id]) /
                video_meta.width
        ) < 0.05:
            track_del_set.add(track_id)

    selected_dets = {}
    for frame, track_and_box in sort_tracks:
        frame_dets = []
        for track_id, box in track_and_box:
            if track_id not in track_del_set:
                frame_dets.append(box)
        selected_dets[frame] = frame_dets

    # Add back the balls lost by sort
    for frame in ball_dets:
        for box in ball_dets[frame]:
            if box._id not in tracked_box_ids:
                if frame not in selected_dets:
                    selected_dets[frame] = []
                selected_dets[frame].append(box)
    return selected_dets


def track_using_sort_and_racket_agreement(
    anchor_dets, racket_dets, video_meta
):
    sort_tracks = compute_sort_tracks(anchor_dets)

    track_del_set = set()
    conflict_sets = set()
    track_racket_count = Counter()
    for frame, track_and_box in sort_tracks:
        rackets = racket_dets[frame]
        for track_id, box in track_and_box:
            for r in rackets:
                if box.iou(r) > 0:
                    track_racket_count[track_id] += r.score

        if len(track_and_box) > 2:
            conflict_sets.add(tuple(sorted(a[0] for a in track_and_box)))

    for conflict in conflict_sets:
        # Resolve the conflicts by taking the tracks with the most
        # racket agreements
        conflict = sorted(conflict, key=lambda x: -track_racket_count[x])
        for i in range(2, len(conflict)):
            track_del_set.add(conflict[i])
    print('    deleting {} / {} tracks due to conflicts'.format(
        len(track_del_set), len(track_racket_count)))

    selected_dets = {}
    for frame, track_and_box in sort_tracks:
        frame_dets = []
        for track_id, box in track_and_box:
            if track_id not in track_del_set:
                frame_dets.append(box)
        selected_dets[frame] = frame_dets
    return selected_dets


def filter_sort_tracks(
        tournament, tracked_anchors, person_dets, plausible_balls, video_meta
):
    tracked_anchors_wo_size_limits = track_bboxes_naive(
        tournament, tracked_anchors, person_dets,
        video_meta, enforce_size_limits=False)

    sort_tracks = compute_sort_tracks(tracked_anchors_wo_size_limits)

    MAX_PLAYER_HEIGHT_PX = \
        MAX_PLAYER_HEIGHT_BY_TOURNAMENT[tournament] * video_meta.height

    track_del_set = set()
    track_min_frame = defaultdict(lambda: 0xFFFFFFFF)
    track_max_frame = defaultdict(lambda: -1)
    track_frame_count = Counter()
    for frame, track_and_box in sort_tracks:
        for track_id, box in track_and_box:
            track_frame_count[track_id] += 1
            track_min_frame[track_id] = min(track_min_frame[track_id], frame)
            track_max_frame[track_id] = max(track_max_frame[track_id], frame)
            if box.h > MAX_PLAYER_HEIGHT_PX:
                track_del_set.add(track_id)

    del_ranges = [(track_min_frame[track_id], track_max_frame[track_id])
                  for track_id in track_del_set]
    print('    deleting {} / {} tracks that contain an oversize person'.format(
        len(track_del_set), len(track_frame_count)))

    # Tracks that overlap a ball
    has_plausible_ball = set()
    for frame in plausible_balls:
        if len(plausible_balls[frame]) > 0:
            for track_id in track_min_frame:
                if (track_id not in has_plausible_ball
                    and between(frame, track_min_frame[track_id],
                                track_max_frame[track_id])):
                    has_plausible_ball.add(track_id)
    print('    deleting {} / {} tracks that do not overlap a ball'.format(
        len(track_frame_count) - len(has_plausible_ball),
        len(track_frame_count)))

    selected_dets = {}
    for frame, track_and_box in sort_tracks:
        for a, b in del_ranges:
            if between(frame, a, b):
                break
        else:
            frame_dets = []
            for track_id, box in track_and_box:
                if track_id in has_plausible_ball:
                    frame_dets.append(box)
            selected_dets[frame] = frame_dets
    return selected_dets


def filter_by_vertical_separation(tracked_anchors, video_meta):
    sort_tracks = compute_sort_tracks(tracked_anchors)

    # Tracks that overlap at least one other track for some amount of time
    HAS_NEIGHBOR_FRAC = 0.25
    track_frame_count = Counter()
    neighbor_count = Counter()
    for frame, track_and_box in sort_tracks:
        for track_id, box in track_and_box:
            track_frame_count[track_id] += 1
            if len(track_and_box) > 1:
                neighbor_count[track_id] += 1
    has_enough_neighbors = {
        track_id for track_id in neighbor_count
        if neighbor_count[track_id] / track_frame_count[track_id]
            >= HAS_NEIGHBOR_FRAC}
    print('    deleting {} / {} tracks that lack consistent neighbors'.format(
        len(track_frame_count) - len(has_enough_neighbors),
        len(track_frame_count)))

    # Tracks that lack vertical separation
    #   - The players should not start out with similar y coordinates
    #   - The players should not always remain in the same y band
    track_max_vsep = defaultdict(float)
    track_init_vsep = {}
    for frame, track_and_box in sort_tracks:
        if len(track_and_box) > 1:
            for track_id, box in track_and_box:
                for track_id2, box2 in track_and_box:
                    if track_id < track_id2:
                        vsep = abs(box.cy - box2.cy) / video_meta.height
                        track_max_vsep[track_id] = max(
                            track_max_vsep[track_id], vsep)
                        track_max_vsep[track_id2] = max(
                            track_max_vsep[track_id2], vsep)
                        if track_id not in track_init_vsep:
                            track_init_vsep[track_id] = vsep
                        if track_id2 not in track_init_vsep:
                            track_init_vsep[track_id2] = vsep

    MIN_MAX_VSEP = 0.33
    MIN_INIT_VSEP = 0.17
    has_enough_vsep = {
        track_id for track_id in track_max_vsep
        if track_max_vsep[track_id] >= MIN_MAX_VSEP
        and track_init_vsep[track_id] >= MIN_INIT_VSEP}
    print('    deleting {} / {} tracks that lack vertical separation'.format(
        len(track_max_vsep) - len(has_enough_vsep), len(track_max_vsep)))

    selected_dets = {}
    for frame, track_and_box in sort_tracks:
        frame_dets = []
        for track_id, box in track_and_box:
            if (track_id in has_enough_vsep
                and track_id in has_enough_neighbors
            ):
                frame_dets.append(box)
        selected_dets[frame] = frame_dets
    return selected_dets


def find_one_person_locations(anchor_dets, min_consecutive=25):
    cands = []
    for frame in sorted(anchor_dets):
        if len(anchor_dets[frame]) == 1:
            cands.append(frame)

    cand_locs = []
    num_consecutive = 0
    prev = None
    for frame in cands:
        if frame - 1 == prev:
            num_consecutive += 1
            if num_consecutive == min_consecutive:
                cand_locs.append(frame)
        else:
            num_consecutive = 0
        prev = frame
    return cand_locs


def interpolate_missing_bboxes(anchor_dets, max_gap=3):
    interpolated_boxes = defaultdict(list)
    for frame in sorted(anchor_dets):
        curr_frame_boxes = anchor_dets[frame]
        next_frame_boxes = anchor_dets.get(frame + 1, [])
        for a in curr_frame_boxes:
            has_next = False
            for b in next_frame_boxes:
                if b.iou(a) > 0:
                    has_next = True

            if not has_next:
                # Search for next bbox
                for i in range(1, max_gap + 1):
                    cand_frame_boxes = anchor_dets.get(frame + i, [])
                    best = None
                    best_iou = 0.1
                    for b in cand_frame_boxes:
                        iou = b.iou(a)
                        if iou > best_iou:
                            best = b
                            best_iou = iou
                    if best is not None:
                        print('    gap:', frame, '--', frame + i)
                        for j in range(1, i):
                            interpolated_boxes[frame + j].append(a.union(b))
                        break

    for frame in interpolated_boxes:
        if frame not in anchor_dets:
            anchor_dets[frame] = []
        for b in interpolated_boxes[frame]:
            anchor_dets[frame].append(b)
    return anchor_dets


def filter_by_screen_position(anchor_dets, video_meta):

    def get_contiguous_dets(dets, min_len=2):
        intervals = []
        start = None
        prev_frame = -100
        for frame, frame_dets in sorted(dets.items()):
            if len(frame_dets) == 0:
                continue

            if frame - 1 != prev_frame:
                if start is not None:
                    if prev_frame - start > min_len:
                        intervals.append((start, prev_frame))
                start = frame
            prev_frame = frame

        if prev_frame - start > min_len:
            intervals.append((start, prev_frame))
        return intervals

    # We do not want all of the tracks in the upper or lower part of the frame
    del_intervals = []
    person_intervals = get_contiguous_dets(anchor_dets)
    for a, b in person_intervals:
        for frame in range(a, b + 1):
            frame_dets = anchor_dets[frame]
            if len(frame_dets) >= 2:
                upper_count = 0
                lower_count = 0
                for box in frame_dets:
                    if box.cy / video_meta.height < 0.33:
                        upper_count += 1
                    elif box.cy / video_meta.height > 0.67:
                        lower_count += 1
                if (upper_count == len(frame_dets)
                    or lower_count == len(frame_dets)
                ):
                    del_intervals.append((a, b))
                    break

    print('    deleting {} / {} intervals based on screen positions'.format(
        len(del_intervals), len(person_intervals)))
    for a, b in del_intervals:
        for i in range(a, b + 1):
            del anchor_dets[i]
    return anchor_dets


def collect_rackets(person_dets, racket_dets):
    selected_rackets = defaultdict(list)
    for frame in racket_dets:
        for r in racket_dets[frame]:
            for p in person_dets.get(frame, []):
                if r.iou(p.expand(1.1)) > 0:
                    selected_rackets[frame].append(r)
                    break
    return selected_rackets


def convert_detections_to_py(obj_dets):
    label_to_frame_dets = defaultdict(lambda: defaultdict(list))
    for frame, dets in obj_dets:
        for cls, score, bbox in dets:
            if cls in COCO_FILTER_SET:
                box = Box(bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2,
                          bbox[2], bbox[3], score, cls)
                if box.area == 0:
                    print('Ignored box w/ area 0!')
                label_to_frame_dets[cls][frame].append(box)
    return label_to_frame_dets


def convert_detections_to_json(dets):
    result = []
    for frame in sorted(dets):
        frame_dets = dets[frame]
        if len(frame_dets) > 0:
            result.append((
                frame, [
                    {'score': box.score, 'xywh': [box.x, box.y, box.w, box.h]}
                    for box in sorted(frame_dets, key=lambda x: -x.score)]
            ))
    return result


def save_tracking_results(out_file, person_det, racket_det, ball_det):
    store_gz_json(out_file, {
        'person': convert_detections_to_json(person_det),
        'racket': convert_detections_to_json(racket_det),
        'ball': convert_detections_to_json(ball_det),
    }, indent=2)


def process_video(tournament, obj_det_file, frame_diff_file, video_file,
                  out_file):
    global CURR_VIDEO_FILE
    CURR_VIDEO_FILE = video_file

    print('Detections:', obj_det_file)
    print('Video:', video_file)

    if video_file is None:
        # Dummy it up; note, unable to visualize
        video_meta = DUMMY_VIDEO_META
    else:
        video_meta = get_metadata(video_file)
    obj_dets = convert_detections_to_py(load_gz_json(obj_det_file))

    # Estimate likely hard cuts from frame diffs
    print('  Finding hard cuts')
    hard_cuts = find_hard_cuts(load_gz_json(frame_diff_file))
    # FIXME: SORT tracker does not receive boundary information

    # Filter out static balls (mostly false positives)
    print('  Filtering implausible balls with SORT')
    plausible_balls = filter_implausible_balls_with_sort(
        obj_dets[COCO_NAMES_INV['sports ball']], video_meta)

    # Find likely candidates for the players as anchors
    #   - exclude frames with only a few people
    #   - exclude frames with large people
    #   - person near racket
    print('  Finding potential anchors')
    potential_anchors = find_potential_anchors(
        tournament, obj_dets, video_meta)

    person_dets = obj_dets[COCO_NAMES_INV['person']]

    # Track anchors naively (bidirectionally)
    print('  Tracking naively (1st pass)')
    tracked_anchors = track_bboxes_naive(
        tournament, potential_anchors, person_dets,
        video_meta, boundaries=hard_cuts)

    # Keep only the tracks that have high agreement with rackets
    print('  Tracking with SORT and checking racket agreement')
    tracked_anchors = track_using_sort_and_racket_agreement(
        tracked_anchors, obj_dets[COCO_NAMES_INV['tennis racket']],
        video_meta)

    # Filter tracks:
    #   - Remove frames where the anchors size changed too much
    #     (e.g., due to not being the right camera)
    #   - Remove tracks with no temporal overlap with a plausible ball
    #   - Remove tracks that never overlap another in time (need 2 players)
    print('  Tracking with SORT and filtering with heuristics')
    tracked_anchors = filter_sort_tracks(
        tournament, tracked_anchors, person_dets,
        plausible_balls, video_meta)

    # Track anchors naively (bidirectionally)
    print('  Tracking naively with lower IoU (2nd pass)')
    tracked_anchors = track_bboxes_naive(
        tournament, tracked_anchors, person_dets,
        video_meta, min_iou=0.25, boundaries=hard_cuts)

    # Interpolate anchors when bboxes are missing
    print('  Interpolating gaps between tracked bboxes')
    tracked_anchors = interpolate_missing_bboxes(tracked_anchors)

    # Filter tracks where the two tracks lack vertical seperation
    #   - This may be due to the wrong camera
    print('  Tracking with SORT and filtering tracks w/o vertical separation')
    tracked_anchors = filter_by_vertical_separation(
        tracked_anchors, video_meta)

    # Filter frames where all tracks end up in the top/bottom of the frame
    tracked_anchors = filter_by_screen_position(tracked_anchors, video_meta)

    print('  Collecting rackets near tracked players')
    collected_rackets = collect_rackets(
        tracked_anchors, obj_dets[COCO_NAMES_INV['tennis racket']])

    # display_boxes(CURR_VIDEO_FILE, tracked_anchors,
    #               obj_dets[COCO_NAMES_INV['person']])

    # Find scenes where only one person is tracked (these are probably mistakes)
    # one_person_locs = find_one_person_locations(tracked_anchors)
    # print(len(one_person_locs))
    # display_boxes(CURR_VIDEO_FILE, tracked_anchors, plausible_balls,
    #               frame_num=one_person_locs[0] - 100)

    if out_file is not None:
        print('Output:', out_file)
        save_tracking_results(
            out_file, tracked_anchors, collected_rackets, plausible_balls)


def process_video_star(args):
    process_video(*args)


def get_tournament(fname):
    return fname.rsplit('/', 1)[-1].split('_', 1)[0]


def get_video_name(fname):
    return fname.rsplit('/', 1)[-1].split('.', 1)[0]


def main(obj_det_file, frame_diff_file, video_file, out_file,
         overwrite, parallelism):

    def get_video_file(fname):
        if video_file is None:
            return None
        elif os.path.isdir(video_file):
            video_name = get_video_name(fname)
            return os.path.join(video_file, video_name + '.mp4')
        else:
            return video_file

    def get_out_file(fname):
        if out_file is None:
            return None
        elif os.path.isdir(out_file) or os.path.isdir(obj_det_file):
            os.makedirs(out_file, exist_ok=True)
            video_name = get_video_name(fname)
            return os.path.join(out_file, video_name + '.track.json.gz')
        else:
            return out_file

    def get_frame_diff_file(fname):
        if os.path.isdir(frame_diff_file):
            video_name = get_video_name(fname)
            return os.path.join(frame_diff_file, video_name + '.diff.json.gz')
        else:
            return frame_diff_file

    if os.path.isdir(obj_det_file):
        worker_args = []
        for curr_obj_file in sorted(os.listdir(obj_det_file)):
            curr_obj_path = os.path.join(obj_det_file, curr_obj_file)
            curr_video_file = get_video_file(curr_obj_file)
            curr_diff_file = get_frame_diff_file(curr_obj_file)

            curr_out_file = get_out_file(curr_obj_file)
            if (not overwrite and curr_out_file is not None
                and os.path.exists(curr_out_file)
            ):
                print('Output already exists:', out_file)
                continue

            worker_args.append((
                get_tournament(curr_obj_file), curr_obj_path,
                curr_diff_file, curr_video_file, curr_out_file))
        with Pool(parallelism) as p:
            for _ in tqdm(p.imap_unordered(
                process_video_star, worker_args), total=len(worker_args)
            ):
                pass
    else:
        curr_video_file = get_video_file(obj_det_file)
        curr_diff_file = get_frame_diff_file(obj_det_file)
        curr_out_file = get_out_file(obj_det_file)
        if (not overwrite and curr_out_file is not None
            and os.path.exists(curr_out_file)
        ):
            print('Output already exists:', curr_out_file)
        else:
            process_video(
                get_tournament(obj_det_file), obj_det_file, curr_diff_file,
                curr_video_file, curr_out_file)


if __name__ == '__main__':
    main(**vars(get_args()))