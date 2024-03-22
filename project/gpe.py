import cv2
import numpy as np


def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


def same_cluster(image, i, j, cluster, ratio=1):
    neighbors = [(i - ratio, j - ratio), (i - ratio, j), (i - ratio, j + ratio), (i, j - ratio), (i, j + ratio),
                 (i + ratio, j - ratio), (i + ratio, j), (i + ratio, j + ratio)]
    for neighbor in neighbors:
        if neighbor[0] >= 0 and neighbor[0] < image.shape[0] and neighbor[1] >= 0 and neighbor[1] < image.shape[1]:
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
    return len(clusters)


if __name__ == "__main__":
    img = cv2.imread('44.jpg')
    # resize image
    ratio = img.shape[1] / img.shape[0]
    height = 800
    width = int(height * ratio)
    print(width, height)

    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img = cv2.medianBlur(img, 15)
    # img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)

    edges = cv2.dilate(edges, None, iterations=10)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(img)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    result = cv2.bitwise_and(img, result)

    # result = cv2.GaussianBlur(result, (41, 41), sigmaX=0)

    # Detect how many pieces are in the image
    print(db_scan(result))

    cv2.imshow('Original Image', img)
    cv2.imshow('Edges', edges)
    cv2.imshow('Filled Contours', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
