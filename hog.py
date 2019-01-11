import cv2
from skimage.feature import hog
import numpy as np


class HOG:
    def __init__(self, window_size=(256, 256)):
        # new_size = (32, 32)
        self.wnd_size = window_size
        self.pixels_per_cell = (8, 8)
        self.orientations = 9
        self.hog_size = (
            self.wnd_size[0] // self.pixels_per_cell[0],  # rows
            self.wnd_size[1] // self.pixels_per_cell[1],  # cols
            self.orientations
        )
        self.hann = self._cosine_window()

    def _cosine_window(self):
        w = h = self.hog_size[0]
        c = np.sqrt(w * h) / 3
        result = np.zeros(self.hog_size)
        mid_x = w // 2
        mid_y = h // 2

        for x in range(w):
            for y in range(h):
                elem = np.exp(-((mid_x - x) ** 2 / (2 * c ** 2) +
                                (mid_y - y) ** 2 / (2 * c ** 2)))
                for k in range(self.hog_size[2]):
                    result[x, y, k] = elem

        return result

    def compute(self, image):
        image = cv2.resize(image, self.wnd_size)
        h = hog(image, orientations=self.orientations, feature_vector=False,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=(1, 1), visualize=False, multichannel=True)

        new_shape = np.squeeze(h).shape
        h = np.reshape(h, new_shape)
        h = np.multiply(h, self.hann)

        return h
