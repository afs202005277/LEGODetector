import cv2
import numpy as np
import matplotlib.pyplot as plt

TARGET_WIDTH = 944
TARGET_HEIGHT = 1133


def get_blob_params():
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 5
    params.maxThreshold = 220

    params.filterByArea = True
    params.minArea = 750  # You may need to adjust this based on the size of your Lego pieces
    params.maxArea = 10000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 0  # 0 => blobs darker than the background; 255 => blobs ligther than background (pelo q percebi)
    return params


def image_setup(image_name):
    original = cv2.imread(image_name)
    original = cv2.resize(original, (TARGET_HEIGHT, TARGET_WIDTH))
    original = cv2.GaussianBlur(original, (5, 5), 0)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return original, gray


def main():
    original, gray = image_setup('t2_55.jpg')
    params = get_blob_params()
    if cv2.__version__.startswith('2.'):
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray)
    print(len(keypoints))

    img_key_points = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", img_key_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
