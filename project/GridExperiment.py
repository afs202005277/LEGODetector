import final


def evaluate_function(image, params):
    # 2. Resize the image

    image = final.resize_image(image)
    
    original_image = image.copy()

    without_background, contours = final.background_removal(image, iterations=6)
    without_background, _ = final.image_segmentation(without_background, contours, original_image)
    clusters = final.db_scan(without_background)
    colors_hue = {
    "red": 5,
    "orange": params['orange'],
    "yellow": 35,
    "lime": params['lime'],
    "green": 70,
    "turquoise": params['turquoise'],
    "cyan": 100,
    "coral": 110,
    "blue": 125,
    "purple": 135,
    "magenta": params['magenta'],
    "pink": 180,
}
    pieces, colors = final.color_scan(clusters, without_background, params['minpoints'], colors_hue)
    return colors
