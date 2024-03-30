import time

import cv2
import sys
import numpy as np
from utils import display_images, resize_image

BB_MIN_HEIGHT = 200
BB_MIN_WIDTH = 200

CLIP_LIMIT_CLAHE = 2.0
TILE_GRID_SIZE_CLAHE = (8, 8)

BILATERAL_FILTER_D = 5
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75

CANNY_THRESHOLD1 = 60
CANNY_THRESHOLD2 = 100
DILATE_ITERATIONS = 10


def improve_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.medianBlur(gray_img, 11)
    blurred_img = cv2.GaussianBlur(blurred_img, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT_CLAHE, tileGridSize=TILE_GRID_SIZE_CLAHE
    )
    enhanced_img = clahe.apply(blurred_img)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_img = cv2.filter2D(enhanced_img, -1, kernel)

    blur_img = cv2.bilateralFilter(
        sharp_img,
        BILATERAL_FILTER_D,
        BILATERAL_FILTER_SIGMA_COLOR,
        BILATERAL_FILTER_SIGMA_SPACE,
    )

    return blur_img


def find_edges(enhanced_img):
    edges = cv2.Canny(enhanced_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    edges = cv2.dilate(edges, None, iterations=DILATE_ITERATIONS)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = cv2.drawContours(
        np.zeros_like(enhanced_img), contours, -1, (255, 255, 255), thickness=cv2.FILLED
    )
    return edges, contours, contours_img


def draw_bbs(img, bbs):
    img_clone = img.copy()
    for bb in bbs:
        x, y, w, h = bb
        cv2.rectangle(img_clone, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return img_clone


def get_bb(contours):
    bb = [cv2.boundingRect(contour) for contour in contours]
    clean_bb = []
    for i in range(len(bb)):
        # Check if the bounding box is not too small
        (x, y, w, h) = bb[i]

        if w < BB_MIN_WIDTH or h < BB_MIN_HEIGHT:
            continue

        # Check if the bounding box is not contained in another bounding box
        is_contained = False
        for j in range(len(bb)):
            if i != j:
                (x2, y2, w2, h2) = bb[j]
                if x2 <= x and y2 <= y and x2 + w2 >= x + w and y2 + h2 >= y + h:
                    is_contained = True
                    break

        if not is_contained:
            clean_bb.append(bb[i])

    return clean_bb


def daniel(img, display=True):

    enhanced_img = improve_img(img)
    edges, contours, contours_img = find_edges(enhanced_img)
    bb = get_bb(contours)
    img_bb = draw_bbs(img, bb)

    if display:
        # display_images(
        #     [img, enhanced_img, edges, contours_img, img_bb],
        #     [
        #         "Original Image",
        #         "Enhanced Image",
        #         "Edges",
        #         "Contours Image",
        #         "Bounding Boxes",
        #     ],
        #     (600, 800),
        # )
        display_images(
            [enhanced_img, contours_img, edges],
            [
                "Enhanced Image",
                "Contours",
                "Edges",
            ],
            (600, 800),
        )

    # return len(bb), 0, id_execution
    return len(bb), 0


if __name__ == "__main__":
    path_to_img = sys.argv[1]
    num_pieces = daniel(cv2.imread(path_to_img))
    print(f"Number of pieces: {num_pieces}")
