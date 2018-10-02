from tracker import BlockMatching
import cv2
import os
import numpy as np

from trackers import Trackers

img = cv2.imread('tc_Ball_ce2\\0001.jpg')
bbox = (344+10, 339, 344 + 40, 339 + 43-10)

# img = cv2.imread('frames\\0.png')
# bbox = (0, 40, 20, 60)

# img = cv2.imread('bag\\00000001.jpg')
# bbox = (334, 128, 438, 189)
trackers = Trackers(img, bbox)

for i in range(2, len(os.listdir(os.path.join(os.path.curdir, 'tc_Ball_ce2')))):
    # cv_img = cv2.imread('bag\\' + str(i).zfill(8) + '.jpg')
    cv_img = cv2.imread('tc_Ball_ce2\\' + str(i).zfill(4) + '.jpg')
    # cv_img = cv2.imread('frames\\' + str(i) + '.png')

    trackers.update(cv_img)
    trackers.update_weights(*trackers.find_bdv())
    x, y = trackers.find_bdv()
    trackers.check_transition(cv_img, x, y)
    trackers.draw_trackers(cv_img)

    bbox = (bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y)
    trackers.reinit(cv_img, bbox)

    cv2.rectangle(cv_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color=(0, 0, 255))
    cv2.imshow('fsg', cv_img)
    cv2.imwrite('{}.png'.format(i), cv_img)
    cv2.waitKey()
