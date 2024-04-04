import cv2
import final
import numpy as np



def evaluate_function(image, params):
    # 2. Resize the image
    image = final.resize_image(image)

    original_image = image.copy()

    without_background, contours = final.background_removal(image)
    without_background = final.image_segmentation(without_background, contours, original_image)
    clusters = final.db_scan(without_background)
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
    pieces, colors = final.color_scan(clusters, without_background, params['minpoints'], colors_hue)
    return colors
