import cv2
import numpy as np
import sys
from utils import display_images


def hough_method(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray, 100, 150, apertureSize=3)
    detected_lines = cv2.HoughLines(edge_image, 1, np.pi / 180, 200)
    
    for line in detected_lines:
        for rho, theta in line:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x_0 = cos_theta * rho
            y_0 = sin_theta * rho
            x_1 = int(x_0 + 1000 * (-sin_theta))
            y_1 = int(y_0 + 1000 * (cos_theta))
            x_2 = int(x_0 - 1000 * (-sin_theta))
            y_2 = int(y_0 - 1000 * (cos_theta))

            cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 5)

    display_images([img, edge_image], ["Image with lines", "Edge Image"], (800, 600))


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    #img_blurred = cv2.GaussianBlur(img, (9, 9), 0)
    lines = hough_method(img)
