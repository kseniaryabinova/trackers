import cv2
import numpy as np

from hog import HOG
from utils import show


class KCF:
    def __init__(self, image, coords):
        """
        :param image: initial image with ROI
        :param coords: [x1, y1, x2, y2]
        """
        self._lambda = 0.0001
        self._sigma = 0.6
        self.image_size = image.shape
        self._interp_factor = 0.012
        self._hog = HOG()

        self._roi = coords
        sub_wnd_coords = self._make_subwindow()
        self._sub_wnd_size = (sub_wnd_coords[2] - sub_wnd_coords[0],
                              sub_wnd_coords[3] - sub_wnd_coords[1])
        self._sub_wnd_coords = sub_wnd_coords

        self._y = self._make_y_values(self._hog.hog_size[0],
                                      self._hog.hog_size[1])

        self._x_hog = self._hog.compute(image[
                                        sub_wnd_coords[0]:sub_wnd_coords[2],
                                        sub_wnd_coords[1]:sub_wnd_coords[3]
                                        ])

        self._alpha = self._train(self._x_hog, True)

    def _make_y_values(self, w, h):
        def rearrange(arr):
            mid_x = w // 2
            mid_y = h // 2

            res = np.zeros((w, h), dtype=np.float32)
            res[mid_x:mid_x * 2, mid_y:mid_y * 2] = arr[0:mid_x, 0:mid_y]
            res[0:mid_x, 0:mid_y] = arr[mid_x:mid_x * 2, mid_y:mid_y * 2]
            res[0:mid_x, mid_y:mid_y * 2] = arr[mid_x:mid_x * 2, 0:mid_y]
            res[mid_x:mid_x * 2, 0:mid_y] = arr[0:mid_x, mid_y:mid_y * 2]

            return res

        c = np.sqrt(w * h) / 9

        result = np.zeros((w, h), dtype=np.float32)

        mid_x = w // 2
        mid_y = h // 2

        for x in range(w):
            for y in range(h):
                result[x, y] = np.exp(-((mid_x - x) ** 2 / (2 * c ** 2) +
                                        (mid_y - y) ** 2 / (2 * c ** 2)))

        result = np.fft.fft(result) / result.shape[1]
        return result

    def _make_subwindow(self):
        """
        возвращает координаты subwindow
        :return: [x1, y1, x2, y2]
        """
        margin = 0.75
        w = margin * (self._roi[2] - self._roi[0])
        h = margin * (self._roi[3] - self._roi[1])

        sub_wnd_coords = [self._roi[0] - w, self._roi[1] - h,
                          self._roi[2] + w, self._roi[3] + h]

        # если выходит за границы изображения:
        if sub_wnd_coords[0] <= 0:
            sub_wnd_coords[0] = 1
        if sub_wnd_coords[1] <= 0:
            sub_wnd_coords[1] = 1
        if sub_wnd_coords[2] >= self.image_size[1]:
            sub_wnd_coords[2] = self.image_size[1] - 1
        if sub_wnd_coords[3] >= self.image_size[0]:
            sub_wnd_coords[3] = self.image_size[0] - 1

        return int(sub_wnd_coords[0]), int(sub_wnd_coords[1]), \
               int(sub_wnd_coords[2]), int(sub_wnd_coords[3])

    def update(self, image):
        new_roi = self._detect(image)
        self._roi = (int(new_roi[0]), int(new_roi[1]),
                     int(new_roi[0] + (self._roi[2] - self._roi[0])),
                     int(new_roi[1] + (self._roi[3] - self._roi[1])))
        sub_wnd_coords = self._make_subwindow()
        self._sub_wnd_size = (sub_wnd_coords[2] - sub_wnd_coords[0],
                              sub_wnd_coords[3] - sub_wnd_coords[1])
        self._sub_wnd_coords = sub_wnd_coords

        self._x_hog = self._hog.compute(
            image[sub_wnd_coords[0]:sub_wnd_coords[2],
            sub_wnd_coords[1]:sub_wnd_coords[3]])

        self._alpha = self._train(self._x_hog, False)
        return self._roi

    def _train(self, new_hog, is_first_time):
        k = self._find_gaussian_kernel(new_hog, new_hog)
        alpha = self._y / (np.fft.fft(k) / k.shape[1] + self._lambda)

        show(np.real(k), 'train_kernel')
        show(np.real(alpha), 'train_alpha')

        if is_first_time:
            return alpha
        return (1 - self._interp_factor) * self._alpha + \
               self._interp_factor * alpha

    def _detect(self, image):
        # FIXME какой-то баг с корреляцией
        z_coords = self._make_subwindow()
        z = image[z_coords[0]:z_coords[2], z_coords[1]:z_coords[3]]
        z_hog = self._hog.compute(z)

        k_xz = self._find_gaussian_kernel(self._x_hog, z_hog)
        f_z = np.fft.ifft(np.multiply(np.fft.fft(k_xz), self._alpha))

        show(np.real(k_xz), '_detect_k_xz')
        show(np.real(f_z), '_detect_correlation')

        _, max_val, _, max_loc = cv2.minMaxLoc(f_z.real)
        print(max_val, max_loc)
        return self._get_coords(max_loc)

    def _get_coords(self, max_loc):
        hog_row = max_loc[1] // self._hog.hog_size[0]
        hog_col = max_loc[0] % self._hog.hog_size[1]

        scale_row = hog_row * self._hog.pixels_per_cell[0] + \
                    self._hog.pixels_per_cell[0] // 2
        scale_col = hog_col * self._hog.pixels_per_cell[1] + \
                    self._hog.pixels_per_cell[1] // 2

        sub_wnd_row = (self._hog.wnd_size[0]+(self._sub_wnd_size[0]))/scale_row
        sub_wnd_col = (self._hog.wnd_size[1]+(self._sub_wnd_size[1]))/scale_col

        sub_wnd_row -= (self._roi[2] - self._roi[0]) / 4
        sub_wnd_col -= (self._roi[3] - self._roi[1]) / 4

        return self._sub_wnd_coords[0] + sub_wnd_row, \
               self._sub_wnd_coords[1] + sub_wnd_col

    def _find_gaussian_kernel(self, x, y):
        correlation = np.zeros([self._hog.hog_size[0], self._hog.hog_size[1]],
                               dtype=np.complex)
        for i in range(self._hog.hog_size[2]):
            x_fft = np.fft.fft(x[:, :, i])
            y_fft = np.fft.fft(y[:, :, i])
            result = np.multiply(np.conj(x_fft), y_fft)
            correlation = np.add(correlation, result)

        norm1 = np.multiply(x, x).sum()
        norm2 = np.multiply(y, y).sum()
        result = (norm1 + norm2 - 2 * np.real(np.fft.ifft(correlation))) / \
                 (self._hog.hog_size[0] * self._hog.hog_size[1] * self._hog.hog_size[2])

        return np.exp(-1 / (self._sigma ** 2) * result)
