import cv2
from tracker import KCF


if __name__ == "__main__":
    img = cv2.imread('img.png')
    tracker = KCF(img, (200, 200, 300, 300))

    for _ in range(30):
        img = cv2.imread('img.png')
        bbox = tracker.update(img)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color=(255, 0, 0), thickness=5)
        cv2.imshow('1', img)
        cv2.waitKey(0)
