from tracker import BlockMatching
import cv2
import os
import numpy as np

from trackers import Trackers

# path = '/media/ksenia/39e992da-a01a-4384-a580-e798bb2aab2a/datasets/cfnet-validation/tc_Ball_ce2'
# path = '/media/ksenia/39e992da-a01a-4384-a580-e798bb2aab2a/datasets/cfnet-validation/vot2014_torus'
path = '/media/ksenia/39e992da-a01a-4384-a580-e798bb2aab2a/datasets/cfnet-validation/vot2014_hand1'

# img = cv2.imread(os.path.join(path, '0001.jpg'))
# bbox = (344+10, 339, 344 + 40, 339 + 43-10)

img =  cv2.imread(os.path.join(path, '00000001.jpg'))
# bbox = (168, 80, 217, 130)
bbox = (225, 92, 268, 137)

# img = cv2.imread('bag\\00000001.jpg')
# bbox = (334, 128, 438, 189)
trackers = Trackers(img, bbox)

for i in range(2, len(os.listdir(path))):
    cv_img = cv2.imread(os.path.join(path, str(i).zfill(8) + '.jpg'))
    # cv_img = cv2.imread(os.path.join(path, str(i).zfill(4) + '.jpg'))

    trackers.update(cv_img)
    x, y = trackers.find_bdv()
    trackers.update_weights(x, y)
    trackers.check_transition(cv_img, x, y)
    trackers.draw_trackers(cv_img)

    bbox = (bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y)
    trackers.reinit(cv_img, bbox, x, y)

    cv2.rectangle(cv_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color=(0, 0, 255))
    cv2.imshow('fsg', cv_img)
    cv2.imwrite('{}.png'.format(i), cv_img)
    cv2.waitKey()
