import cv2
import numpy as np

img = cv2.imread('44.jpg')

#resize image
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_image[:,:,2] = np.clip(hsv_image[:,:,2] * 1.5, 0, 255).astype(np.uint8)
hsv_image[:,:,1] = np.clip(hsv_image[:,:,1] + 9, 0, 255).astype(np.uint8)


img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150)

edges = cv2.dilate(edges, None, iterations=4)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = np.zeros_like(img)
for contour in contours:
    if cv2.contourArea(contour) > 0:
        print(contour)
        cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)


result = cv2.bitwise_and(img, result)



cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges)
cv2.imshow('Filled Contours', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
