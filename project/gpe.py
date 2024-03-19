import cv2
import numpy as np

def is_black(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0

def same_cluster(image, i, j, cluster):
    neighbors = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
    for neighbor in neighbors:
        if neighbor[0] >= 0 and neighbor[0] < image.shape[0] and neighbor[1] >= 0 and neighbor[1] < image.shape[1]:
            if neighbor in cluster:
                return True
    return False

def merge_clusters(image, clusters):
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for pixel in clusters[j]:
                if same_cluster(image, pixel[0], pixel[1], clusters[i]):
                    for ele in clusters[j]:
                        clusters[i].append(ele)
                    clusters.pop(j)
                    return clusters
    return clusters

def clear_clusters(image, clusters):
    temp = -1
    while temp != len(clusters):
        temp = len(clusters)
        clusters = merge_clusters(image, clusters)  
    return clusters

def db_scan(image):
    clusters = []
    #For every pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #If the pixel is not black
            if not is_black(image[i][j]):
                found = False
                for cluster in clusters:
                    if same_cluster(image, i, j, cluster):
                        cluster.append((i, j))
                        found = True
                        break
                if not found:
                    clusters.append([(i, j)])
                    clusters = clear_clusters(image, clusters)       

    clusters = clear_clusters(image, clusters)  
    return len(clusters)

                
img = cv2.imread('44.jpg')

#resize image
scale_percent = 5 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_image[:,:,2] = np.clip(hsv_image[:,:,2] * 1.5, 0, 255).astype(np.uint8)
hsv_image[:,:,1] = np.clip(hsv_image[:,:,1] + 9, 0, 255).astype(np.uint8)


img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

img = cv2.medianBlur(img, 5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150)

edges = cv2.dilate(edges, None, iterations=4)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = np.zeros_like(img)
for contour in contours:
    if cv2.contourArea(contour) > 0:
        cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)


result = cv2.bitwise_and(img, result)

#result = cv2.GaussianBlur(result, (41, 41), sigmaX=0)

#Detect how many pieces are in the image
print(db_scan(result))



cv2.imshow('Original Image', img)
cv2.imshow('Filled Contours', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
