import cv2
import gpe
import numpy as np


def remove_background_canny(image, params):

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + 12, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 125)

    edges = cv2.dilate(edges, None, iterations=6)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result


def evaluate_function(image, params):
    without_background = remove_background_canny(image, params)
    clusters = gpe.db_scan(without_background)
    bg_color = gpe.get_bg_color(image, without_background)
    pieces, colors = gpe.color_scan(clusters, without_background, bg_color, params['threshold1'], params['threshold2'], params['threshold3'], params['minpoints'])
    return colors
