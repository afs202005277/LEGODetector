import json
import sys
import cv2
import numpy as np

from utils import equalize_hist_wrapper
import math

SAME_COLOR_THRESHOLD = 110
SAME_COLOR_THRESHOLD2 = 40
SAME_COLOR_THRESHOLD3 = 50
MIN_POINTS_COLOR = 0.26
MIN_POINTS_COLOR_BGR = 50
VALUE = 60

colors_hue = {
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
        if (
                neighbor[0] >= 0
                and neighbor[0] < image.shape[0]
                and neighbor[1] >= 0
                and neighbor[1] < image.shape[1]
        ):
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
    Perform Density-Based Spatial Clustering of Applications with Noise (DBSCAN) on the image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        list: List of clusters.
"""


def db_scan(image):
    clusters = []
    ratio = image.shape[0] // 75
    # For every pixel in the image
    for i in range(0, image.shape[0], ratio):
        for j in range(0, image.shape[1], ratio):
            # If the pixel is not black
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


def color_scan(clusters, image, min_points_color=MIN_POINTS_COLOR, colors_hue=colors_hue):
    # image = cv2.GaussianBlur(image, (41, 41), 0)

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
                    if v >= VALUE:
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


def image_segmentation(image, contours, original_image):
    bbs = [cv2.boundingRect(contour) for contour in contours]
    combination = (15, 160, 100)
    masks = []
    original_image = equalize_hist_wrapper(original_image, *combination)
    mask = np.zeros(original_image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    for bb_idx in range(len(bbs)):
        bb = bbs[bb_idx]
        cv2.setRNGSeed(0)
        (mask, bg_model, fgModel) = cv2.grabCut(original_image, mask, bb, bg_model, fg_model, 8,
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
    height = 800
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
    result, bbs = image_segmentation(result, contours, original_image)

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

    #print(num_colors)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    process_images(sys.argv[1], sys.argv[2])
