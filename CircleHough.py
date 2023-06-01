import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# lire l image
img = cv.imread("mort.png", cv.IMREAD_GRAYSCALE)

img = img[:, 100:1250]


## Gaussian Blur
img_blur = cv.GaussianBlur(img, (7, 7), 0)

# Appliquer un seuillage adaptatif
thresholded = cv.adaptiveThreshold(
    img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
)

# Appliquer l'égalisation d'histogramme
equalized = cv.equalizeHist(thresholded)

# Appliquer la détection de contours (Filtre de Canny)
edges = cv.Canny(equalized, 30, 100)

# Apply the Hough Circle Transform
circles = cv.HoughCircles(
    img_blur,
    cv.HOUGH_GRADIENT,
    1.1,
    minDist=48,
    param1=270,
    param2=11.5,
    minRadius=31,
    maxRadius=45,
)
circles_ = np.array(circles)
shape = circles_.shape
print(shape[1])
# Check if circles were found
if circles is not None:
    # Convert the (x, y) center coordinates and radius to integers
    circles = np.round(circles[0, :]).astype(int)
    # Draw the detected circles on the original image
    for x, y, r in circles:
        cv.circle(img, (x, y), r, (0, 0, 255), 2)

    # Display the image with circles
    plt.imshow(img)
    plt.title("Nombre de cercles est : {}".format(shape[1]))
    plt.show()

else:
    print("No circles were found.")
