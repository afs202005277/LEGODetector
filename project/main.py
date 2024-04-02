import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import display_images
import gpe

DISPLAY = False
TARGET_WIDTH = 944
TARGET_HEIGHT = 1133


def remove_background_canny_v3(image_path):
    image = cv2.imread(image_path)

    # resize image
    ratio = image.shape[1] / image.shape[0]
    height = 800
    width = int(height * ratio)

    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] + 12, 0, 255)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 125)

    edges = cv2.dilate(edges, None, iterations=10)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return image, result, contours


def remove_background_canny_v2(image_path):
    image = cv2.imread(image_path)

    # resize image
    ratio = image.shape[1] / image.shape[0]
    height = 800
    width = int(height * ratio)

    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] + 12, 0, 255)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 125)

    edges = cv2.dilate(edges, None, iterations=10)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return image, result, contours


def remove_background_canny(image_path):
    image = cv2.imread(image_path)

    # resize image
    ratio = image.shape[1] / image.shape[0]
    height, width = 0, 0
    if ratio > 1:
        width = 800
        height = int(width / ratio)
    else:
        height = 800
        width = int(height * ratio)

    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)

    edges = cv2.dilate(edges, None, iterations=10)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result


def get_blob_params():
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 5
    params.maxThreshold = 220

    params.filterByArea = True
    params.minArea = (
        750  # You may need to adjust this based on the size of your Lego pieces
    )
    params.maxArea = 10000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 0  # 0 => blobs darker than the background; 255 => blobs ligther than background (pelo q percebi)
    return params


def remove_background(image_path, hue_margin=30, sat_margin=100, val_margin=255):
    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()
    image = cv2.GaussianBlur(image, (55, 55), sigmaX=0)

    if image is None:
        print("Error: Unable to read image.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Flatten the image to a 2D array (height * width, 3)
    flattened_image = hsv_image.reshape((-1, 3))

    # Compute the most common color using k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        np.float32(flattened_image), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    most_common_color = np.uint8(centers[0])
    # Define the range for background color with a margin of error
    lower_bound = np.array(
        [
            most_common_color[0] - hue_margin,
            most_common_color[1] - sat_margin,
            most_common_color[2] - val_margin,
        ]
    )
    upper_bound = np.array(
        [
            most_common_color[0] + hue_margin,
            most_common_color[1] + sat_margin,
            most_common_color[2] + val_margin,
        ]
    )

    # Create a mask for pixels within the defined color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Invert the mask to represent the background
    background_mask = cv2.bitwise_not(mask)

    # Replace background pixels with original pixels
    result = cv2.bitwise_and(original, original, mask=background_mask)

    # Display the result
    if DISPLAY:
        display_images([original, result], ["Original Image", "Background Removed"], (800, 600))

    return result


def show_hue(image):
    image = image.copy()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the hue channel
    hue_channel = hsv_image[:, :, 0]

    # Display the hue channel as a grayscale image
    if DISPLAY:
        cv2.imshow("Hue Channel", cv2.medianBlur(hue_channel, 5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def image_setup(image_name):
    original = cv2.imread(image_name)
    original = cv2.resize(original, (TARGET_HEIGHT, TARGET_WIDTH))
    # show_hue(original)
    original = cv2.bilateralFilter(original, 11, 75, 75)
    # original = cv2.GaussianBlur(original, (5, 5), 0)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])
    sharpening_kernel = sharpening_kernel / np.sum(sharpening_kernel)
    gray = cv2.filter2D(gray, -1, sharpening_kernel)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return original, gray


def blob_detection(gray_image):
    params = get_blob_params()
    if cv2.__version__.startswith("2."):
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray_image)

    img_key_points = cv2.drawKeypoints(
        gray_image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if DISPLAY:
        display_images([img_key_points], ['Keypoints'])
    return len(keypoints)


def color_detection(image):
    image = cv2.bilateralFilter(image, 9, 75, 75)

    # Convert BGR image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for each Lego color in HSV space
    color_thresholds = {
        "red": ([0, 100, 100], [10, 255, 255]),
        "orange": ([11, 100, 100], [20, 255, 255]),
        "yellow": ([21, 100, 100], [30, 255, 255]),
        "green": ([31, 100, 100], [70, 255, 255]),
        "blue": ([90, 100, 100], [130, 255, 255]),
        "purple": ([131, 100, 100], [160, 255, 255]),
        "white": ([0, 0, 200], [180, 50, 255]),
        "black": ([0, 0, 0], [180, 255, 100]),
    }

    # Initialize an empty set to store distinct colors
    distinct_colors = set()

    # Iterate over each color threshold
    for color_name, (lower, upper) in color_thresholds.items():
        # Create a mask using the color thresholds
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # If any contours are found, consider it as one instance of the color
        if contours:
            distinct_colors.add(color_name)

    if DISPLAY:
        display_images([image], ["Image with Contours"])

    return len(distinct_colors)


def make_image_brighter(image, factor):
    image = image.copy()

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into channels
    h, s, v = cv2.split(hsv_image)

    # Multiply the value channel by the factor
    v_multiplied = np.clip(v * factor, 0, 255).astype(np.uint8)

    # Merge the channels back into an HSV image
    hsv_image_modified = cv2.merge([h, s, v_multiplied])

    # Convert the modified HSV image back to BGR color space
    modified_image = cv2.cvtColor(hsv_image_modified, cv2.COLOR_HSV2BGR)

    # Display the original and modified images
    if DISPLAY:
        cv2.imshow("Original Image", image)
        cv2.imshow("Modified Image", modified_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return modified_image


def detect_pieces_v1(filename):
    _, gray = image_setup(filename)
    return blob_detection(gray)


def detect_pieces_v2(filename):
    without_background = remove_background(filename)
    gray_wo_bgr = cv2.cvtColor(without_background, cv2.COLOR_BGR2GRAY)
    return blob_detection(cv2.equalizeHist(gray_wo_bgr))


def detect_pieces_v3(filename):
    without_background = remove_background_canny(filename)
    return len(gpe.db_scan(without_background))


def detect_pieces_v4(filename):
    image, without_background, _ = remove_background_canny_v2(filename)
    clusters = gpe.db_scan(without_background)
    bg_color = gpe.get_bg_color(image, without_background)
    pieces, colors = gpe.color_scan(clusters, without_background, bg_color)
    return pieces


def detect_pieces_v5(filename):
    image, without_background, contours = remove_background_canny_v2(filename)
    without_background = gpe.andre(without_background, contours, image)
    clusters = gpe.db_scan(without_background)
    bg_color = gpe.get_bg_color(image, without_background)
    pieces, colors = gpe.color_scan(clusters, without_background, bg_color)
    return pieces


def count_colors_v1(filename):
    original, _ = image_setup(filename)
    return color_detection(original)


def count_colors_v2(filename):
    without_background = remove_background(filename)
    return color_detection(without_background)


def count_colors_v3(filename):
    image, without_background, _ = remove_background_canny_v2(filename)
    clusters = gpe.db_scan(without_background)
    bg_color = gpe.get_bg_color(image, without_background)
    pieces, colors = gpe.color_scan(clusters, without_background, bg_color)
    return colors


def count_colors_v4(filename):
    image, without_background, contours = remove_background_canny_v2(filename)
    without_background = gpe.andre(without_background, contours, image)
    clusters = gpe.db_scan(without_background)
    bg_color = gpe.get_bg_color(image, without_background)
    pieces, colors = gpe.color_scan(clusters, without_background, bg_color)
    return colors


def main(filename):
    # print(detect_pieces_v1(filename))
    # print(detect_pieces_v2(filename))
    start = time.time()
    print(detect_pieces_v4(filename))
    print("Time taken: ", time.time() - start, "seconds")
    # print(count_colors_v1(filename))
    # print(count_colors_v1(filename))


if __name__ == "__main__":
    main('44.jpg')
