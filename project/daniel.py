import cv2
import sys
import numpy as np
from utils import display_images

BB_MIN_HEIGHT = 200
BB_MIN_WIDTH = 200

def improve_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(270, 270))
    enhanced_img = clahe.apply(gray_img)

    enhanced_img = cv2.medianBlur(enhanced_img, 15)
    enhanced_img = cv2.GaussianBlur(enhanced_img, (7, 7), 0)

    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)

    return enhanced_img


def find_edges(enhanced_img):
    edges = cv2.Canny(enhanced_img, 50, 70)
    # kernel = np.ones((15, 15), np.uint8)
    edges = cv2.dilate(edges, None, iterations=10)
    # edges = cv2.erode(edges, kernel, iterations=4)

    # edges = cv2.GaussianBlur(edges, (5, 5), 0)
    edges = cv2.medianBlur(edges, 5)

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


def daniel(path_to_img, display=False):
    img = cv2.imread(path_to_img)
    enhanced_img = improve_img(img)
    edges, contours, contours_img = find_edges(enhanced_img)
    bb = get_bb(contours)
    img_bb = draw_bbs(img, bb)

    # if display:
    #    display_images(
    #        [img, img_bb, enhanced_img, contours_img, edges],
    #        [
    #            "Original Image",
    #            "Bounding Boxes",
    #            "Enhanced Image",
    #            "Contours Image",
    #            "Edges",
    #        ],
    #        (600, 800),
    #    )

    if display:
        display_images(
            [img_bb],
            [
                "Bounding Boxes",
            ],
            (600, 800),
        )

    return len(bb)


if __name__ == "__main__":
    path_to_img = sys.argv[1]
    num_pieces = daniel(path_to_img, display=True)
    print(f"Number of pieces: {num_pieces}")
