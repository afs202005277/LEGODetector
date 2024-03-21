import cv2
import gpe
import numpy as np


def remove_background_canny(image, params):
    image = cv2.medianBlur(image, params['median_blur'])
    image = cv2.GaussianBlur(image, (params['gaussian_blur'], params['gaussian_blur']), sigmaX=params['sigma'])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params['canny_min'], params['canny_max'])

    edges = cv2.dilate(edges, None, iterations=params['dilation_it'])

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result


def evaluate_function(image, params):
    without_background = remove_background_canny(image, params)
    return gpe.db_scan(without_background)
