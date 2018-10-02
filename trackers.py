from copy import copy

import cv2
import numpy as np

from tracker import BlockMatching, State


class Trackers:
    def __init__(self, frame, bbox):
        self.trackers = []
        self.cur_vectors = {}
        self.vectors = []
        self.new_rects = []
        self.rectangles = []
        self.debug_color = []

        self.create_trackers(frame, bbox)
        for tracker in self.trackers:
            tracker.state = State.INLIER
        self.init_trackers_amount = len(self.trackers)

    def create_trackers(self, frame, bbox):
        small_w = (bbox[2] - bbox[0]) // 7
        small_h = (bbox[3] - bbox[1]) // 7
        big_w = (bbox[2] - bbox[0]) // 4
        big_h = (bbox[3] - bbox[1]) // 4
        for x1 in range(bbox[0], bbox[2] - small_w, small_w):
            for y1 in range(bbox[1], bbox[3] - small_h, small_h):
                tracker = BlockMatching(frame,
                                        (x1, y1, x1 + small_w, y1 + small_h))
                self.trackers.append(tracker)
        for x1 in range(bbox[0], bbox[2] - big_w, big_w):
            for y1 in range(bbox[1], bbox[3] - big_h, big_h):
                tracker = BlockMatching(frame, (x1, y1, x1 + big_w, y1 + big_h))
                self.trackers.append(tracker)
        for i in range(len(self.trackers) - 1, -1, -1):
            if self.trackers[i].state == State.OUTLIER:
                self.trackers.pop(i)

    def find_bdv(self):
        self.cur_vectors = {}
        for tracker in self.trackers:
            if tracker.state == State.INLIER:
                if tracker.cur_vector not in self.cur_vectors.keys():
                    self.cur_vectors[tracker.cur_vector] = tracker.weight
                else:
                    self.cur_vectors[tracker.cur_vector] += tracker.weight
        print(self.cur_vectors)
        print(max(self.cur_vectors, key=self.cur_vectors.get))
        return max(self.cur_vectors, key=self.cur_vectors.get)

    def update_weights(self, x, y):
        delta = 7
        for j in range(len(self.trackers) - 1, -1, -1):
            if self.trackers[j].state == State.INLIER:
                length = max(abs(self.trackers[j].cur_vector[0] - x),
                             abs(self.trackers[j].cur_vector[1] - y))
                if length >= delta:
                    self.trackers.pop(j)
                else:
                    if length >= 1:
                        self.trackers[j].weight *= (delta - length) / (
                                    delta)
                    else:
                        self.trackers[j].weight += 1

    def check_transition(self, frame, x, y):
        delta = 7
        for j in range(len(self.trackers) - 1, -1, -1):
            if self.trackers[j].state == State.TRANSITION:
                v_x, v_y, _, _ = self.trackers[j].get_motion_vector(frame)
                length = max(abs(v_x - x), abs(v_y - y))
                if length >= delta:
                    self.trackers[j].state = State.OUTLIER
                elif length >= 1:
                    self.trackers[j].weight *= (delta - length) / (delta)
                else:
                    self.trackers[j].weight += 1
                    self.trackers[j].transition_count += 1

    def update(self, frame):
        self.vectors = []
        self.rectangles = []
        self.new_rects = []
        self.debug_color = []
        tracker_iter = 0
        for tracker in self.trackers:
            if tracker.state == State.TRANSITION and tracker.transition_count >= 2:
                tracker.state = State.INLIER
                tracker.transition_count = 0
            if tracker.state == State.INLIER:
                self.rectangles.append(((tracker.cur_bbox[0], tracker.cur_bbox[1]),
                                        (tracker.cur_bbox[2], tracker.cur_bbox[3])))
                v_x, v_y, p1, p2 = tracker.get_motion_vector(frame)
                self.new_rects.append((p1, p2))
                self.vectors.append((v_x, v_y))
                self.debug_color.append(100 * tracker.weight)
                print(tracker_iter, len(self.trackers))
                tracker_iter += 1

    def draw_trackers(self, frame):
        for rectangle, new_rect, color, vector in zip(self.rectangles, self.new_rects,
                                            self.debug_color, self.vectors):
            cv2.rectangle(frame, *rectangle, color=(255, color, 0))
            # cv2.rectangle(frame, *new_rect, color=(255, color, 0))
            x = (rectangle[1][0] + rectangle[0][0]) // 2
            y = (rectangle[1][1] + rectangle[0][1]) // 2
            cv2.line(frame, (x, y), (x + vector[0], y + vector[1]),
                     color=(0, 0, 255))

    def reinit(self, frame, bbox):
        _amount = 0
        for tracker in self.trackers:
            if tracker.state in [State.INLIER, State.TRANSITION]:
                _amount += 1
        print('\t{} {} {}'.format(_amount, self.init_trackers_amount,
                                  _amount / self.init_trackers_amount))
        if _amount / self.init_trackers_amount <= 0.25:
            self.create_trackers(frame, bbox)
