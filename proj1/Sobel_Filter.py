import numpy as np
from scipy.signal import convolve2d
from skimage import io, img_as_float32, color
import matplotlib.pyplot as plt

def Convolve2D(image, kernel):

    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the padding width
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Prepare the output image
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Perform the convolution operation
            output[i, j] = np.sum(region * kernel)
    
    return output

# Load the image and convert it to float32
I = img_as_float32(io.imread('../images/squirrel.jpg'))  # Load as grayscale

# Convert the image to grayscale if it's not already
if I.ndim == 3:  # Check if the image is RGB
    I_gray = color.rgb2gray(I)  # Convert to grayscale

# Define the Sobel kernel (example for the x-direction)
sobelKernel = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# Perform 2D convolution
I_Convolve2D = Convolve2D(I_gray, sobelKernel)
I_full = convolve2d(I_gray, sobelKernel, mode='full')
I_same = convolve2d(I_gray, sobelKernel, mode='same')
I_valid = convolve2d(I_gray, sobelKernel, mode='valid')

# Create a figure and a set of subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12, 5))

ax1.imshow(I_gray, cmap='gray') 
ax1.set_title('Gray image')
ax1.axis('off')

ax2.imshow(I_Convolve2D, cmap='gray') 
ax2.set_title('Manual convolution')
ax2.axis('off')

ax3.imshow(I_full, cmap='gray')
ax3.set_title('full mode')
ax3.axis('off')

ax4.imshow(I_same, cmap='gray')
ax4.set_title('same mode')
ax4.axis('off')

ax5.imshow(I_valid, cmap='gray')
ax5.set_title('valid mode')
ax5.axis('off')

# Show the images
plt.tight_layout()
plt.show()

