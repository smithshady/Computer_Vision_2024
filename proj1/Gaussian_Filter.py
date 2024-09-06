import numpy as np
from skimage import io, img_as_float32, color
from scipy.ndimage import gaussian_filter, shift
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Load the image and convert it to float32
I = img_as_float32(io.imread('mines-logo.jpg'))  # Load as grayscale

# Convert the image to grayscale if it's not already
if I.ndim == 3:  # Check if the image is RGB
    I_gray = color.rgb2gray(I)  # Convert to grayscale

# Define the Gaussian filter kernel size and sigma
sigma = 5  # Standard deviation for Gaussian kernel

# Apply Gaussian filter to create a blurred/shadow image
blurred_image = gaussian_filter(I_gray, sigma=sigma)

# Define the shift amount for the shadow effect
shift_amount = (20, 20)  # Pixels to shift left-down

# Apply shift filter to move the shadow
shifted_blurred_image = shift(blurred_image, shift=shift_amount, mode='nearest')

# Overlap the images by summing pixel values
output_image = np.clip(I_gray + shifted_blurred_image, 0, 1)  # Ensure pixel values are in [0, 1]

# Create Gaussian filter kernel image
size = int(6 * sigma + 1)  # Kernel size
x, y = np.meshgrid(np.linspace(-sigma, sigma, size), np.linspace(-sigma, sigma, size))
gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gaussian_kernel /= gaussian_kernel.sum()

# Create a shifted Gaussian kernel for visualization
shifted_kernel = np.roll(gaussian_kernel, shift_amount, axis=(0, 1))

# Display the results
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 6))

# Display the Gaussian filter kernel
ax1.imshow(gaussian_kernel, cmap='gray')
ax1.set_title('Gaussian Kernel')
ax1.axis('off')

# Display the shifted Gaussian kernel
ax2.imshow(shifted_kernel, cmap='gray')
ax2.set_title('Shifted Kernel')
ax2.axis('off')

# Display the blurred image
ax3.imshow(blurred_image, cmap='gray')
ax3.set_title('Blurred Image')
ax3.axis('off')

# Display the shifted blurred image
ax4.imshow(shifted_blurred_image, cmap='gray')
ax4.set_title('Shifted Blurred Image')
ax4.axis('off')

# Display the output image
ax5.imshow(output_image, cmap='gray')
ax5.set_title('Output Image')
ax5.axis('off')

plt.show()
