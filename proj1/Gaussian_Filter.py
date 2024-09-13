import numpy as np
from skimage import io, img_as_float32, color
from scipy.ndimage import gaussian_filter, shift
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Function to create an image with text
def create_image(text, img):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 200)  # Try to load a system font
    bbox = draw.textbbox((0, 0), text, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    draw.text((x, y), text, font=font, fill=text_color)

# Define size
width, height = 800, 400
background_color = (255, 255, 255)  # White background
text_color = (0, 0, 0)  # Black text

# Create a new image with text
img = Image.new('RGB', (width, height), color=background_color)
create_image("MINES", img)

# Convert the image to grayscale and to a NumPy array
I_gray = np.array(img.convert('L'))  # Convert to grayscale and NumPy array

# Apply Gaussian filter to create a blurred/shadow image
sigma = 10
I_blur = gaussian_filter(I_gray, sigma=sigma)

# Define the shift amount for the shadow effect
shift_amount = (20, -20)  # Pixels to shift left-down

# Apply shift filter to move the shadow
I_blur_shift = shift(I_blur, shift=shift_amount, mode='nearest')

# Overlap the images by summing pixel values

#adding the images together, needed weights?
alpha = .7 # original image 
beta = 1 - alpha  # blurred image
I_overlay = np.clip(I_gray*alpha + I_blur_shift*beta, 0, 255)  # Ensure pixel values are in [0, 255]

# Create Gaussian filter kernel image
size = int(6 * sigma + 1)  # Kernel size
x, y = np.meshgrid(np.linspace(-sigma, sigma, size), np.linspace(-sigma, sigma, size))
gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gaussian_kernel /= gaussian_kernel.sum()

# Create a shifted Gaussian kernel for visualization
shifted_kernel = np.roll(gaussian_kernel, shift_amount, axis=(0, 1))

# Display the results
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12, 5))

# Display the gray image
ax1.imshow(I_gray, cmap='gray')
ax1.set_title('Gray Image')
ax1.axis('off')

# Display the Gaussian filter kernel
ax2.imshow(gaussian_kernel, cmap='gray')
ax2.set_title('Gaussian Kernel')
ax2.axis('off')

# Display the shifted Gaussian kernel
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
