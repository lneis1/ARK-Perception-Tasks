import cv2
import numpy as np

# Load the image
image = cv2.imread('D:/Samarth/Clg Extra Stuff/Sem02/ARK/Luna/table.png')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=11)
gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=11)


gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


threshold = 2350000
edges = (gradient_magnitude > threshold) * 255


edges = edges.astype(np.uint8)


cv2.imwrite('edge.png', edges)