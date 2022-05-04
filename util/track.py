from abc import abstractmethod
import numpy as np

from util.box import Box


class AbstractTrackAnnotator:

    @abstractmethod
    def get(self, frame, pose_data):
        raise NotImplementedError()


class NullTrackAnnotator(AbstractTrackAnnotator):

    def get(self, _, pose_data):
        return [x for x in pose_data if x.get('id', -1) >= 0]


class SingleTrackAnnotator(AbstractTrackAnnotator):

    def get(self, _, pose_data):
        assert len(pose_data) == 1
        pose_data[0]['id'] = 0
        return pose_data


class TennisTrackAnnotator(AbstractTrackAnnotator):

    def __init__(self, intervals, trim=0):
        self.interval_dict = {}
        for i, (a, b) in enumerate(intervals):
            for j in range(a + trim, b + 1 - trim):
                self.interval_dict[j] = i

    def get(self, frame, pose_data):
        interval_id = self.interval_dict.get(frame)
        ret = []
        if interval_id is not None:
            bg, fg = TennisTrackAnnotator.select_two_players(pose_data)
            if bg is not None:
                bg['id'] = 2 * interval_id
                ret.append(bg)
            if fg is not None:
                fg['id'] = 2 * interval_id + 1
                ret.append(fg)
        return ret

    VIDEO_HEIGHT = 1080

    @staticmethod
    def select_two_players(all_players):
        assert len(all_players) > 0

        bg, fg = None, None
        if len(all_players) == 1:
            box = Box(*all_players[0]['xywh'])
            if box.cy > TennisTrackAnnotator.VIDEO_HEIGHT // 2:
                fg = all_players[0]
            else:
                bg = all_players[0]
        elif len(all_players) == 2:
            p1, p2 = all_players
            b1 = Box(*p1['xywh'])
            b2 = Box(*p2['xywh'])
            if b1.cy > b2.cy:
                fg = p1
                bg = p2
            else:
                fg = p2
                bg = p1
        else:
            below_mid = []
            above_mid = []
            for p in all_players:
                box = Box(*p['xywh'])
                if box.cy > TennisTrackAnnotator.VIDEO_HEIGHT // 2:
                    below_mid.append((box, p))
                else:
                    above_mid.append((box, p))

            # Let FG be the highest in below mid
            fg_box = None
            for b, p in below_mid:
                if fg is None or fg_box.y2 > b.y2:
                    fg = p
                    fg_box = b

            # Let BG be the lowest in above mid
            bg_box = None
            for b, p in above_mid:
                if bg is None or bg_box.y2 < b.y2:
                    bg = p
                    bg_box = b

        return bg, fg


def find_hard_cuts(frame_diffs, percentile=99.5):
    THRESHOLD = np.percentile(frame_diffs, percentile)

    boundaries = set()
    for i, diff in enumerate(frame_diffs):
        if diff > THRESHOLD:
            boundaries.add(i)   # Cut between (i - 1, i)
    return boundaries


class ShotSingleTrackAnnotator(AbstractTrackAnnotator):

    def __init__(self, pose_dict, frame_diff_array):
        hard_cuts_by_diff = sorted(find_hard_cuts(frame_diff_array))

        hard_cuts_by_bbox = []
        prev_frame = None
        for frame in sorted(pose_dict):
            if prev_frame is not None:
                if frame == prev_frame + 1:
                    prev = Box(*pose_dict[prev_frame][0]['xywh'])
                    curr = Box(*pose_dict[frame][0]['xywh'])
                    if curr.iou(prev) < 0.01:
                        hard_cuts_by_bbox.append(frame)
                else:
                    hard_cuts_by_bbox.append(frame)
            prev_frame = frame

        # TODO: generate track id ranges
        assert 0

    def get(self, frame, pose_data):
        assert len(pose_data) == 1
        pose_data[0]['id'] = 0
        return pose_data
