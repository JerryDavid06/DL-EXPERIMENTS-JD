import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image (change path accordingly)
img = cv2.imread("C:\Users\Jerry David\Downloads\WhatsApp Image 2026-03-10 at 7.12.01 PM.jpeg")   # <-- Put your image file here

# Check if image loaded
if img is None:
    print("Error: Image not found. Check file path.")
    exit()

# Convert BGR to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image to 2D array of pixels
pixels = rgb_img.reshape((-1, 3))
pixels = np.float32(pixels)

# K-means criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Number of clusters
K = 3

# Apply K-means
_, labels, centers = cv2.kmeans(
    pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

# Convert centers to uint8
centers = np.uint8(centers)

# Map labels to segmented image
segmented_img = centers[labels.flatten()]
segmented_img = segmented_img.reshape(rgb_img.shape)

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('Segmented Image (K-Means)')
plt.axis('off')

plt.tight_layout()
plt.show()
