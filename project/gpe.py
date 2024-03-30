import cv2
import numpy as np
from utils import display_images
import math
import os
import random

SAME_COLOR_THRESHOLD = 110
SAME_COLOR_THRESHOLD2 = 40
SAME_COLOR_THRESHOLD3 = 50
MIN_POINTS_COLOR = 50


def get_bg_color(initial_image, image_no_bg):
    bg_color = None
    while True:
        height = random.randint(0, image_no_bg.shape[0] - 1)
        width = random.randint(0, image_no_bg.shape[1] - 1)
        if is_black(image_no_bg[height][width]):
            bg_color = initial_image[height][width]
            break
    return bg_color

def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


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

def clear_colors(colors, threshold=SAME_COLOR_THRESHOLD2):
    temp = -1
    while temp != len(colors):
        temp = len(colors)
        colors = merge_colors(colors, threshold)
    return colors


def clear_clusters(image, clusters, ratio):
    temp = -1
    while temp != len(clusters):
        temp = len(clusters)
        clusters = merge_clusters(image, clusters, ratio)
    return clusters


def db_scan(image):
    clusters = []
    ratio = image.shape[0] // 100
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


def color_scan(clusters, image, bg_color, threshold1=SAME_COLOR_THRESHOLD, threshold2=SAME_COLOR_THRESHOLD2, threshold3=SAME_COLOR_THRESHOLD3, min_points_color=MIN_POINTS_COLOR):
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


        colors_no_bg = remove_bg(colors, bg_color)
        
        for color in colors_no_bg:
            full_colors.append(color)

        c += max(len(colors_no_bg), 1)

    return c, max(len(full_colors), 1)

def color_dist(color1, color2):
    return math.sqrt((int(color1[0]) - int(color2[0])) ** 2 + (int(color1[1]) - int(color2[1])) ** 2 + (int(color1[2]) - int(color2[2])) ** 2)

def remove_bg(colors, bg_color):
    min_dist = 1000000
    min_val = None
    if len(colors) < 2:
        return colors
    for i, color in enumerate(colors):
        if color_dist(color, bg_color) < min_dist:
            min_dist = color_dist(color, bg_color)
            min_val = i
    colors.pop(min_val)
    return colors
         

def points_with_color(color, cluster, image, threshold=SAME_COLOR_THRESHOLD2):
    c = 0
    for point in cluster:
        if color_dist(color, image[point[0]][point[1]]) < threshold:
            c += 1
    return c
        
def grab_cut(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return image * mask2[:, :, np.newaxis]


def draw_bb(img, contours):
    img_clone = img.copy()
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append({"x": x, "y": y, "w": w, "h": h})
        cv2.rectangle(img_clone, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return img_clone, rectangles


if __name__ == "__main__":
    #get a list of every image in the samples folder

    '''samples_folder = "./samples-task1/samples"
    images = []
    for filename in os.listdir(samples_folder):
            images.append(os.path.join(samples_folder, filename))'''



    img = cv2.imread('samples-task1/samples/IMG_20201127_002957.jpg')
    # resize image
    ratio = img.shape[1] / img.shape[0]
    height = 800
    width = int(height * ratio)

    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + 12, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + 3, 0, 255)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    img = cv2.medianBlur(img, 11)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 125)

    edges = cv2.dilate(edges, None, iterations=10)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #img_bb, rectangles = draw_bb(img, contours)
    #display_images([img_bb], ["Bounding Boxes"], (600, 800))

    result = np.zeros_like(img)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(img, result)

    # result = cv2.GaussianBlur(result, (41, 41), sigmaX=0)
    bg_color = get_bg_color(img, result)

    # Detect how many pieces are in the image
    clusters = db_scan(result)
    n = color_scan(clusters, result, bg_color)
    print(n)
    
#grab cut for each rectangle´
    
    '''result2 = np.zeros_like(img)
    for rectangle in rectangles:
        x, y, w, h = rectangle["x"], rectangle["y"], rectangle["w"], rectangle["h"]
        rect = (x, y, w, h)
        grab_cut_img = grab_cut(img, rect)
        result2 = cv2.bitwise_or(result2, grab_cut_img)'''
    cv2.imshow('Original Image', img)
    cv2.imshow('Filled Contours', result)
    #cv2.imshow('Grab Cut', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
