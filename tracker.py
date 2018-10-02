from enum import Enum
from operator import sub
import cv2
import numpy as np


class State(Enum):
    INLIER = 1
    OUTLIER = 2
    TRANSITION = 3


class BlockMatching:
    def __init__(self, frame, bbox: tuple):
        self.state = State.TRANSITION
        self.transition_count = 0
        self.current_cluster = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self.cur_bbox = bbox
        self.cur_vector = (0, 0)
        self.weight = 1

        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.cur_wnd = None
        self.x_offset = None
        self.y_offset = None
        self.update_cur_wnd(frame)
        print(self.x_offset, self.y_offset, self.cur_wnd)

    def update_cur_wnd(self, frame):
        w = int(self.width * 1)
        h = int(self.height * 1)
        self.cur_wnd = [0 if self.cur_bbox[0] - w < 0 else self.cur_bbox[0] - w,
                        0 if self.cur_bbox[1] - h < 0 else self.cur_bbox[1] - h,
                        frame.shape[1] if self.cur_bbox[2] + w > frame.shape[
                            1] else self.cur_bbox[2] + w,
                        frame.shape[0] if self.cur_bbox[3] + h > frame.shape[
                            0] else self.cur_bbox[3] + h]
        self.x_offset = self.cur_wnd[2] - self.cur_wnd[0] - self.width
        self.y_offset = self.cur_wnd[3] - self.cur_wnd[1] - self.height

    def update_weight(self):
        if self.weight >= 2:
            self.weight = 2
        else:
            self.weight += 0.1 / self.weight

    def error_function1(self, possible_cluster):
        error = 0
        for y in range(self.current_cluster.shape[1]):
            for x in range(self.current_cluster.shape[0]):
                pix1 = self.current_cluster[x][y]
                pix2 = possible_cluster[x][y]
                error += abs(pix1[0] - pix2[0]) + abs(pix1[1] - pix2[1]) + \
                         abs(pix1[2] - pix2[2])
        return error

    def error_function(self, possible_cluster):
        error = 0
        for y in range(self.current_cluster.shape[1]):
            for x in range(self.current_cluster.shape[0]):
                pix1 = self.current_cluster[x][y]
                pix2 = possible_cluster[x][y]
                error += np.sqrt((pix1[0] - pix2[0]) ** 2 +
                                 (pix1[1] - pix2[1]) ** 2 +
                                 (pix1[2] - pix2[2]) ** 2)
        return error

    def get_motion_vector(self, frame):
        possibilities = {}
        for y in range(self.cur_wnd[1], self.cur_wnd[1] + self.y_offset):
            for x in range(self.cur_wnd[0], self.cur_wnd[0] + self.x_offset):
                error = self.error_function(
                    frame[y:self.height + y, x:self.width + x])
                possibilities[(x, y)] = error

        v_x, v_y = min(possibilities, key=possibilities.get)
        v_x, v_y = v_x - self.cur_bbox[0], v_y - self.cur_bbox[1]
        self.cur_bbox = (self.cur_bbox[0] + v_x, self.cur_bbox[1] + v_y,
                         self.cur_bbox[2] + v_x, self.cur_bbox[3] + v_y)
        self.update_cur_wnd(frame)
        self.current_cluster = frame[self.cur_bbox[1]:self.cur_bbox[3],
                               self.cur_bbox[0]:self.cur_bbox[2]]
        self.cur_vector = (v_x, v_y)
        return v_x, v_y, (self.cur_bbox[0], self.cur_bbox[1]), \
               (self.cur_bbox[2], self.cur_bbox[3])
