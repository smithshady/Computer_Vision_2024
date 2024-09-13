import numpy as np
from skimage import io, img_as_float32, color
from scipy.ndimage import gaussian_filter, shift
import matplotlib.pyplot as plt

# TODO: need to find a better photo, maybe transparent is what we need?
# TODO: use the actual shift kernel to visualize not make a new one lol

# Load the image and convert it to float32
I = img_as_float32(io.imread('../images/BlasterID.jpg'))

# Convert the image to grayscale if it's in RGB
if I.ndim == 3:
    I_gray = color.rgb2gray(I) 

# Apply Gaussian filter to create a blurred/shadow image
sigma = 10
I_blur = gaussian_filter(I_gray, sigma=sigma)

# Define the shift amount for the shadow effect
shift_amount = (30, 30)  # Pixels to shift left-down

# Apply shift filter to move the shadow
I_blur_shift = shift(I_blur, shift=shift_amount, mode='nearest')

# Overlap the images by summing pixel values
I_overlay = np.clip(I_gray + I_blur_shift, 0, 1)  # Ensure pixel values are in [0, 1]

# Create Gaussian filter kernel image
size = int(6 * sigma + 1)  # Kernel size
x, y = np.meshgrid(np.linspace(-sigma, sigma, size), np.linspace(-sigma, sigma, size))
gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gaussian_kernel /= gaussian_kernel.sum()

# Create a shifted Gaussian kernel for visualization
shifted_kernel = np.roll(gaussian_kernel, shift_amount, axis=(0, 1))

# Display the results
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12, 5))

# Display the Gaussian filter kernel
ax1.imshow(I_gray, cmap='gray')
ax1.set_title('Gray Image')
ax1.axis('off')

# Display the shifted Gaussian kernel
ax2.imshow(gaussian_kernel, cmap='gray')
ax2.set_title('Gaussian Kernel')
ax2.axis('off')

# Display the blurred image
ax3.imshow(shifted_kernel, cmap='gray')
ax3.set_title('Shifted Kernel')
ax3.axis('off')

# Display the shifted blurred image
ax4.imshow(I_blur_shift, cmap='gray')
ax4.set_title('Shifted Blurred Image')
ax4.axis('off')

# Display the output image
ax5.imshow(I_overlay, cmap='gray')
ax5.set_title('Output Image')
ax5.axis('off')

plt.show()
