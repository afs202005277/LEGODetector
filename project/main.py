import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import display_images

DISPLAY = True
TARGET_WIDTH = 944
TARGET_HEIGHT = 1133


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
    print(len(keypoints))

    img_key_points = cv2.drawKeypoints(
        gray_image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imshow("Keypoints", img_key_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    # Display the image with contours
    cv2.imshow("Image with Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


def main(filename):
    # original, gray = image_setup(filename)
    # blob_detection(gray)
    without_background = remove_background(filename)
    # gray_wo_bgr = cv2.cvtColor(without_background, cv2.COLOR_BGR2GRAY)
    # blob_detection(cv2.equalizeHist(gray_wo_bgr))
    # print(color_detection(original))
    # print(color_detection(without_background))
    # remove_background(filename)


if __name__ == "__main__":
    main(sys.argv[1])
