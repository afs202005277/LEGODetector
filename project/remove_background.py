import sys
import cv2
import numpy as np
from utils import display_images


def image_enhancement(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img_clahe = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    img_blur = cv2.GaussianBlur(img_clahe, (9, 9), 0)
    return img_blur

def remove_background(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th_adaptive = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2
    )

    return th_adaptive


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    img_enhanced = image_enhancement(img)
    img_foreground = remove_background(img_enhanced)
    display_images([img_foreground], ["Image without Background"], (800, 600))
