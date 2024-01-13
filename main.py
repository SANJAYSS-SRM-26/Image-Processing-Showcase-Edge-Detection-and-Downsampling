import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# Load the sample image
i = misc.ascent()

# Display the original image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

# Create a copy of the original image for transformation
i_transformed = np.copy(i)

# Get the size of the image
size_x, size_y = i_transformed.shape

# Define a filter for edge detection
filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Set a weight for normalization
weight = 1

# Apply the filter convolution operation
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        output_pixel = np.sum(i[x-1:x+2, y-1:y+2] * filter) * weight
        i_transformed[x, y] = np.clip(output_pixel, 0, 255)

# Display the transformed image after edge detection
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()

# Downsample the image by selecting the maximum pixel value in 2x2 neighborhoods
new_x, new_y = size_x // 2, size_y // 2
newImage = np.zeros((new_x, new_y))

for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = [i_transformed[x:x+2, y:y+2].flatten()]
        newImage[int(x / 2), int(y / 2)] = np.max(pixels)

# Display the downsampled image
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()
