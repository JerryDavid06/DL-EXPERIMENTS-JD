import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/content/image.jpg')

b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((2, 2), np.uint8)

closing = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

sure_bg = cv2.dilate(
    closing,
    kernel,
    iterations=3
)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title("Otsu's Threshold")
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4)
plt.imshow(closing, cmap='gray')
plt.title("MorphologyEx: Closing")
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5)
plt.imshow(sure_bg, cmap='gray')
plt.title("Dilation")
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

plt.imsave('dilation.png', sure_bg)
