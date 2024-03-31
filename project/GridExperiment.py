import cv2
import gpe
import numpy as np


def remove_background_canny(image, params):

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + 12, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, params['median'])
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 125)

    edges = cv2.dilate(edges, None, iterations=params['dilate'])

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result, contours


def evaluate_function(image, params):
    without_background, contours = remove_background_canny(image, params)
    without_background = gpe.andre(without_background, contours, image)
    clusters = gpe.db_scan(without_background)
    colors_hue = {
    "red": params['red'],
    "orange": params['orange'],
    "yellow": 35,
    "lime": 45,
    "green": 70,
    "turquoise": params['turquoise'],
    "cyan": 100,
    "coral": 110,
    "blue": 125,
    "purple": 135,
    "magenta": 155,
    "pink": params['pink'],
}
    pieces, colors = gpe.color_scan(clusters, without_background, params['minpoints'], colors_hue)
    return colors
