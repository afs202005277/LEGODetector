import cv2
import sys
import numpy as np
from utils import display_images


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    # blurred_img = cv2.GaussianBlur(img, (5, 5), 7)
    # blurred_img = cv2.medianBlur(img, 11)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    blurred_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    edges = cv2.Canny(blurred_img, 100, 150)
    edges = cv2.dilate(edges, None, iterations=10)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(img)

    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)
        
    result = cv2.bitwise_and(img, result)

    display_images([result], ["Edges"], (600, 800))
