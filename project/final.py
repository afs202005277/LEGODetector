import json
import sys
import cv2
import numpy as np
import math

SAME_COLOR_THRESHOLD = 110
SAME_COLOR_THRESHOLD2 = 40
SAME_COLOR_THRESHOLD3 = 50
MIN_POINTS_COLOR = 0.26
MIN_POINTS_COLOR_BGR = 50

COLORS_HUE = {
    "red": 5,
    "orange": 19,
    "yellow": 35,
    "lime": 45,
    "green": 70,
    "turquoise": 86,
    "cyan": 100,
    "coral": 110,
    "blue": 125,
    "purple": 135,
    "magenta": 155,
    "pink": 180,
}

"""
    Adjusts the contrast of an image.

    Args:
    - image (numpy.ndarray): Input image in BGR color space.
    - d (int): Diameter of each pixel neighborhood for bilateral filtering.
    - s_color (float): Filter sigma in the color space.
    - s_space (float): Filter sigma in the coordinate space.

    Returns:
    - numpy.ndarray: Image with adjusted contrast.

    If the input image has three channels (BGR), the function converts it to HSV color space, 
    performs histogram equalization on the value channel (V) and inverts it, applies bilateral filtering on the hue channel (H),
    and then converts it back to the BGR color space. 
    If the input image has a single channel, it directly applies histogram equalization on it.

    """


def adjust_image_contrast(image, d, s_color, s_space):
    if image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = 255 - cv2.equalizeHist(v)
        h = cv2.bilateralFilter(h, d, s_color, s_space)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)


"""
Check if a pixel represents the color black.

Args:
    pixel (tuple): A tuple representing the BGR values of a pixel.

Returns:
    bool: True if the pixel represents the color black, False otherwise.
"""


def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


"""
Check if a pixel (i, j) belongs to the same cluster as the given cluster.

Args:
    image (numpy.ndarray): The input image.
    i (int): The row index of the pixel.
    j (int): The column index of the pixel.
    cluster (list): A list of pixels representing a cluster.
    ratio (int, optional): Ratio used for determining neighbors. Defaults to 1.

Returns:
    bool: True if the pixel belongs to the same cluster, False otherwise.
"""


def same_cluster(image, i, j, cluster, ratio=1):
    neighbors = [
        (i - ratio, j - ratio),
        (i - ratio, j),
        (i - ratio, j + ratio),
        (i, j - ratio),
        (i, j + ratio),
        (i + ratio, j - ratio),
        (i + ratio, j),
        (i + ratio, j + ratio),
    ]
    for neighbor in neighbors:
        if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
            if neighbor in cluster:
                return True
    return False


"""
Merge clusters that are adjacent or overlapping.

Args:
    image (numpy.ndarray): The input image.
    clusters (list): List of clusters.
    ratio (int): Ratio used for determining neighbors.

Returns:
    list: List of merged clusters.
"""


def merge_clusters(image, clusters, ratio):
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for pixel in clusters[j]:
                if same_cluster(image, pixel[0], pixel[1], clusters[i], ratio):
                    for ele in clusters[j]:
                        clusters[i].append(ele)
                    clusters.pop(j)
                    return clusters
    return clusters


"""
Merge colors that are very close to each other based on a treshold.

Args:
    colors (list): List of colors.
    threshold (float): treshold used to compare the colors.

Returns:
    list: List of merged colors.
"""


def merge_colors(colors, threshold=SAME_COLOR_THRESHOLD2):
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if color_dist(colors[i], colors[j]) < threshold:
                colors[i][0] = (int(colors[i][0]) + int(colors[j][0])) // 2
                colors[i][1] = (int(colors[i][1]) + int(colors[j][1])) // 2
                colors[i][2] = (int(colors[i][2]) + int(colors[j][2])) // 2
                colors.pop(j)
                return colors
    return colors


"""
Clear clusters by merging adjacent or overlapping clusters.

Args:
    image (numpy.ndarray): The input image.
    clusters (list): List of clusters.
    ratio (int): Ratio used for determining neighbors.

Returns:
    list: List of cleared clusters.
"""


def clear_clusters(image, clusters, ratio):
    temp = -1
    while temp != len(clusters):
        temp = len(clusters)
        clusters = merge_clusters(image, clusters, ratio)
    return clusters


"""
    Perform our version of Density-Based Spatial Clustering of Applications with Noise (DBSCAN) on the image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        list: List of clusters.
"""


def db_scan(image):
    clusters = []
    ratio = image.shape[0] // 75
    for i in range(0, image.shape[0], ratio):
        for j in range(0, image.shape[1], ratio):
            if not is_black(image[i][j]):
                found = False
                for cluster in clusters:
                    if same_cluster(image, i, j, cluster, ratio):
                        cluster.append((i, j))
                        found = True
                        break
                if not found:
                    clusters.append([(i, j)])
                    clusters = clear_clusters(image, clusters, ratio)

    clusters = clear_clusters(image, clusters, ratio)
    return clusters


"""
    Scan clusters and determine their dominant colors.

    Args:
        clusters (list): List of clusters.
        image (numpy.ndarray): The input image.
        min_points_color (int, optional): Minimum points for a color to be considered dominant. Defaults to MIN_POINTS_COLOR.
        colors_hue (dict, optional): Dictionary containing hue values for different colors. Defaults to colors_hue.

    Returns:
        tuple: A tuple containing the number of pieces detected and the number of distinct colors.
    """


def color_scan(clusters, image, min_points_color=MIN_POINTS_COLOR, colors_hue=COLORS_HUE, blur=0):
    if blur != 0:
        image = cv2.medianBlur(image, blur)

    c = 0
    full_colors = set()
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for cluster in clusters:
        colors = []
        colors_dict = {
            "red": [0, 0],
            "orange": [0, 0],
            "yellow": [0, 0],
            "lime": [0, 0],
            "green": [0, 0],
            "turquoise": [0, 0],
            "cyan": [0, 0],
            "coral": [0, 0],
            "blue": [0, 0],
            "purple": [0, 0],
            "magenta": [0, 0],
            "pink": [0, 0],
            "white": [0, 0, 0]
        }
        for x, y in cluster:
            h, s, v = image_hsv[x][y]
            for color in colors_hue:
                if s <= 100:
                    if v >= 180:
                        colors_dict["white"][0] += 1
                    elif v >= 80:
                        colors_dict["white"][1] += 1
                    else:
                        colors_dict["white"][2] += 1
                    break
                if h <= colors_hue[color]:
                    if v >= 60:
                        colors_dict[color][0] += 1
                    else:
                        colors_dict[color][1] += 1
                    break
        for color in colors_dict:
            if color == "white":
                white, gray, black = colors_dict[color]
                if white >= len(cluster) * min_points_color:
                    colors.append("white")
                if gray >= len(cluster) * min_points_color:
                    colors.append("gray")
                if black >= len(cluster) * min_points_color:
                    colors.append("black")
                continue
            bright, dark = colors_dict[color]

            if bright >= len(cluster) * min_points_color:
                colors.append(color)

            if dark >= len(cluster) * min_points_color:
                colors.append(f"dark {color}")

        c += len(colors)

        full_colors.update(colors)

    return c, len(full_colors)


"""
Scan clusters and determine their dominant colors using BGR color space.

Args:
    clusters (list): List of clusters.
    image (numpy.ndarray): The input image.
    threshold1 (float, optional): Threshold for color distance. Defaults to SAME_COLOR_THRESHOLD.
    threshold2 (float, optional): Threshold for color distance for merging. Defaults to SAME_COLOR_THRESHOLD2.
    threshold3 (float, optional): Threshold for color distance for clearing colors. Defaults to SAME_COLOR_THRESHOLD3.
    min_points_color (int, optional): Minimum points for a color to be considered dominant. Defaults to MIN_POINTS_COLOR_BGR.

Returns:
    tuple: A tuple containing the total number of colors detected and the number of distinct colors.
"""


def color_scan_bgr(clusters, image, threshold1=SAME_COLOR_THRESHOLD, threshold2=SAME_COLOR_THRESHOLD2,
                   threshold3=SAME_COLOR_THRESHOLD3, min_points_color=MIN_POINTS_COLOR_BGR):
    c = 0
    full_colors = []
    for cluster in clusters:
        colors = []
        for x, y in cluster:
            color = image[x][y]
            found = False
            for i in range(len(colors)):

                if color_dist(color, colors[i]) < threshold1:
                    found = True
                    colors[i][0] = (int(colors[i][0]) + int(color[0])) // 2
                    colors[i][1] = (int(colors[i][1]) + int(color[1])) // 2
                    colors[i][2] = (int(colors[i][2]) + int(color[2])) // 2
                    break
            if not found:
                colors.append(color)
        colors = clear_colors(colors, threshold2)
        temp = -1
        while temp != len(colors):
            temp = len(colors)
            for i in range(len(colors)):
                if points_with_color(colors[i], cluster, image, threshold2) < min_points_color:
                    colors.pop(i)
                    break

        for color in colors:
            full_colors.append(color)

        c += max(len(colors) - 1, 1)
    full_colors = clear_colors(full_colors, threshold3)

    return c, max(len(full_colors) - 1, 1)


"""
Calculate the Euclidean distance between two colors.

Args:
    color1 (tuple): The first color in BGR format.
    color2 (tuple): The second color in BGR format.

Returns:
    float: The Euclidean distance between the two colors.
"""


def color_dist(color1, color2):
    return math.sqrt((int(color1[0]) - int(color2[0])) ** 2 + (int(color1[1]) - int(color2[1])) ** 2 + (
            int(color1[2]) - int(color2[2])) ** 2)


"""
Count the number of points in a cluster with similar color to a given color.

Args:
    color (tuple): The target color in BGR format.
    cluster (list): List of points representing a cluster.
    image (numpy.ndarray): The input image.
    threshold (float, optional): Threshold for color distance. Defaults to SAME_COLOR_THRESHOLD2.

Returns:
    int: The number of points with similar color in the cluster.
"""


def points_with_color(color, cluster, image, threshold=SAME_COLOR_THRESHOLD2):
    c = 0
    for point in cluster:
        if color_dist(color, image[point[0]][point[1]]) < threshold:
            c += 1
    return c


"""
Clear similar colors in the given list of colors.

Args:
    colors (list): List of colors.
    threshold (float, optional): Threshold for color distance. Defaults to SAME_COLOR_THRESHOLD2.

Returns:
    list: List of cleared colors.
"""


def clear_colors(colors, threshold=SAME_COLOR_THRESHOLD2):
    temp = -1
    while temp != len(colors):
        temp = len(colors)
        colors = merge_colors(colors, threshold)
    return colors


"""
Perform segmentation on the input image using GrabCut with bounding rectangles specified by contours.

Args:
    image (numpy.ndarray): The input image.
    contours (list): List of contours.
    original_image (numpy.ndarray): The original image.

Returns:
    numpy.ndarray: The filtered image after segmentation.
"""


def image_segmentation(image, contours, original_image, its=8):
    bbs = [cv2.boundingRect(contour) for contour in contours]
    combination = (15, 160, 100)
    masks = []
    original_image = adjust_image_contrast(original_image, *combination)
    mask = np.zeros(original_image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    for bb_idx in range(len(bbs)):
        bb = bbs[bb_idx]
        cv2.setRNGSeed(0)
        (mask, bg_model, fgModel) = cv2.grabCut(original_image, mask, bb, bg_model, fg_model, its,
                                                cv2.GC_INIT_WITH_RECT)

        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)

        output_mask = (output_mask * 255).astype("uint8")
        masks.append(output_mask)

        if bb_idx < len(bbs) - 1:
            mask.fill(0)
            bg_model.fill(0)
            fg_model.fill(0)

    if len(masks) == 0:
        return np.zeros_like(image), bbs
    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = cv2.bitwise_or(merged_mask, mask)
    filtered_image = cv2.bitwise_and(image, image, mask=merged_mask)

    return filtered_image, bbs


"""
Perform background removal on the given image using Canny edge detection and contour extraction.

Args:
    image (numpy.ndarray): The input image.

Returns:
    tuple: A tuple containing the resulting image with the background removed and the extracted contours.
"""


def background_removal(image, iterations=10):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] + 12, 0, 255)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] + 3, 0, 255)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)

    edges = cv2.Canny(image, 50, 125)

    edges = cv2.dilate(edges, None, iterations=iterations)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(image)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(image, result)

    return result, contours


"""
Resize the input image while preserving aspect ratio.

Args:
    image (numpy.ndarray): The input image.
    height (int, optional): The target height for resizing. Defaults to 800.

Returns:
    numpy.ndarray: The resized image.
"""


def resize_image(image, height=800):
    ratio = image.shape[1] / image.shape[0]
    width = int(height * ratio)

    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def main(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)

    # 2. Resize the image
    image = resize_image(image)

    original_image = image.copy()

    # 3. Image preprocessing: Background removal
    result, contours = background_removal(image)

    result, _ = image_segmentation(result, contours, original_image)

    # 4. Perform DBSCAN on the image
    clusters = db_scan(result)

    # 5. Scan clusters and determine their dominant colors (Best to find blocks)
    num_blocks, _ = color_scan_bgr(clusters, result)

    # 3. Image preprocessing: Background removal
    result, contours = background_removal(image, iterations=6)
    result, bbs = image_segmentation(result, contours, original_image)

    clusters = db_scan(result)

    # 5. Scan clusters and determine their dominant colors (Best to find colors)
    _, num_colors = color_scan(clusters, result)

    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return num_blocks, num_colors, bbs


def detect_pieces_v1(filename):
    n_blocks, _, _ = main(filename)
    return n_blocks


def count_colors_v1(filename):
    _, n_colors, _ = main(filename)
    return n_colors


def process_images(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []

    for image_file in data["image_files"]:
        print(image_file)
        num_blocks, num_colors, bounding_boxes = main(image_file)
        bounding_boxes = [{"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h} for (x, y, w, h) in bounding_boxes]

        results.append({
            "file_name": image_file,
            "num_colors": num_colors,
            "num_detections": num_blocks,
            "detected_objects": bounding_boxes
        })

    output_data = {"results": results}

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


#################################################################################
#                        Beginning of other approaches tested                   #
#################################################################################

def display_images(images, window_names, dimensions=None):
    named_images = zip(window_names, images)
    for name, image in named_images:
        resized_img = resize_image(image, dimensions)
        cv2.imshow(name, resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

    edges = cv2.Canny(image, 50, 125)

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

    return image, result


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
        750
    )
    params.maxArea = 10000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 0
    return params


def remove_background(image_path, hue_margin=30, sat_margin=100, val_margin=255):
    image = cv2.imread(image_path)
    original = image.copy()
    image = cv2.GaussianBlur(image, (55, 55), sigmaX=0)

    if image is None:
        print("Error: Unable to read image.")
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    flattened_image = hsv_image.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        np.float32(flattened_image), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    most_common_color = np.uint8(centers[0])

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

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    background_mask = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(original, original, mask=background_mask)

    display_images([original, result], ["Original Image", "Background Removed"], (800, 600))

    return result


def show_hue(image):
    image = image.copy()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue_channel = hsv_image[:, :, 0]

    cv2.imshow("Hue Channel", cv2.medianBlur(hue_channel, 5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_setup(image_name):
    original = cv2.imread(image_name)
    original = cv2.resize(original, (1133, 944))
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

    distinct_colors = set()

    for color_name, (lower, upper) in color_thresholds.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        if contours:
            distinct_colors.add(color_name)

    display_images([image], ["Image with Contours"])

    return len(distinct_colors)


def make_image_brighter(image, factor):
    image = image.copy()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    v_multiplied = np.clip(v * factor, 0, 255).astype(np.uint8)

    hsv_image_modified = cv2.merge([h, s, v_multiplied])

    modified_image = cv2.cvtColor(hsv_image_modified, cv2.COLOR_HSV2BGR)

    cv2.imshow("Original Image", image)
    cv2.imshow("Modified Image", modified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return modified_image


def _detect_pieces_v1(filename):
    _, gray = image_setup(filename)
    return blob_detection(gray)


def _detect_pieces_v2(filename):
    without_background = remove_background(filename)
    gray_wo_bgr = cv2.cvtColor(without_background, cv2.COLOR_BGR2GRAY)
    return blob_detection(cv2.equalizeHist(gray_wo_bgr))


def _detect_pieces_v3(filename):
    without_background = remove_background_canny(filename)
    return len(db_scan(without_background))


def _detect_pieces_v4(filename):
    image, without_background = remove_background_canny_v2(filename)
    clusters = db_scan(without_background)
    pieces, colors = color_scan(clusters, without_background)
    return pieces


def _detect_pieces_v5(filename):
    image, without_background, contours = remove_background_canny_v3(filename)
    without_background, _ = image_segmentation(without_background, contours, image)
    clusters = db_scan(without_background)
    pieces, colors = color_scan(clusters, without_background)
    return pieces


def _count_colors_v1(filename):
    original, _ = image_setup(filename)
    return color_detection(original)


def _count_colors_v2(filename):
    without_background = remove_background(filename)
    return color_detection(without_background)


def _count_colors_v3(filename):
    image, without_background = remove_background_canny_v2(filename)
    clusters = db_scan(without_background)
    pieces, colors = color_scan(clusters, without_background)
    return colors


def _count_colors_v4(filename):
    image, without_background, contours = remove_background_canny_v3(filename)
    without_background, _ = image_segmentation(without_background, contours, image)
    clusters = db_scan(without_background)
    pieces, colors = color_scan(clusters, without_background)
    return colors


#################################################################################
#                        End of other approaches tested                         #
#################################################################################

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    process_images(sys.argv[1], sys.argv[2])
