import cv2
import numpy as np
import sys
from utils import display_images
import math


def hough_method(img):
    img_og = img.copy()
    img2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2_canny = cv2.Canny(img2_gray, 50, 200)

    num_votes = 100

    lines = cv2.HoughLines(img2_canny, 1, np.pi / 180, num_votes, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            # Draw the line
            cv2.line(img_og, pt1, pt2, (255, 0, 0), 3)
    
    return img_og

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    lines = hough_method(img)
    display_images([lines], ["Hough"], (600, 800))
