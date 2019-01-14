import cv2
from tracker import KCF
from bindings_to_tests import Tests
from utils import create_video


def test_on_img_png():
    img = cv2.imread('img.png')
    tracker = KCF(img, (200, 200, 300, 300))

    for _ in range(30):
        img = cv2.imread('img.png')
        bbox = tracker.update(img)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color=(255, 0, 0), thickness=5)
        cv2.imshow('1', img)
        cv2.waitKey(0)


def test_on_frames_dir():
    img = cv2.imread('frames/000.png')
    tracker = KCF(img, (200, 200, 300, 300))

    for i in range(1, 30):
        img = cv2.imread('frames/{:03d}.png'.format(i))
        bbox = tracker.update(img)
        print(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color=(255, 0, 0), thickness=5)
        cv2.imshow('1', img)
        cv2.waitKey(0)


def test_cfnet_validation():
    test = Tests("/media/ksenia/39e992da-a01a-4384-a580-e798bb2aab2a/"
                 "datasets/cfnet-validation",
                 "PY_KCF_")
    path = test.try_get_next_file()
    tracker = None

    while path is not None:
        img = cv2.imread(path)

        if test.check_is_new_video():
            coords = test.read_current_groundtruth()
            coords = (coords[0], coords[1],
                      coords[0] + coords[2], coords[1] + coords[3])
            tracker = KCF(img, coords)

        result = tracker.update(img)
        cv2.rectangle(img,
                      (result[0], result[1]),
                      (result[2], result[3]),
                      (0, 0, 255))

        cv2.imshow('1', img)
        if cv2.waitKey(1) == 27:
            break
        # test.bboxes_to_file(*result)

        path = test.try_get_next_file()


if __name__ == "__main__":
    test_cfnet_validation()
