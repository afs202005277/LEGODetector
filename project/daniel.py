import cv2
import sys
import numpy as np
from utils import display_images

def improve_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced_img = gray_img
    enhanced_img = cv2.GaussianBlur(enhanced_img, (15, 15), 0)
    return enhanced_img

def find_edges(enhanced_img):
    edges = cv2.Canny(enhanced_img, 50, 70)
    kernel = np.ones((15, 15), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=4)
    edges = cv2.erode(edges, kernel, iterations=4)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours

def count_pieces(contours):
    return len(contours)

def daniel(path_to_img, display=False):
    img = cv2.imread(path_to_img)
    enhanced_img = improve_img(img)
    edges, contours = find_edges(enhanced_img)
    num_pieces = count_pieces(contours)

    if display:
        display_images(
            [img, enhanced_img, edges],
            ["Original Image", "Enhanced Image", "Edges"],
            (600, 800),
        )
    
    return num_pieces


if __name__ == "__main__":
    path_to_img = sys.argv[1]
    num_pieces = daniel(path_to_img, display=True)
    print(f"Number of pieces: {num_pieces}")