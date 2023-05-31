# hough line transformation
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("mort.png")
image = image[:, 100:1250]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# Apply Canny edge detection to obtain edges
edges = cv2.Canny(gray, 50, 150)

# Perform Hough Line Transform
lines = cv2.HoughLines(edges, rho=0.6, theta=np.pi / 160, threshold=135)
print(lines)
# Draw detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with detected lines
plt.imshow(image)
plt.show()
