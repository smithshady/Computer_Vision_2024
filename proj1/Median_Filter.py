import numpy as np
from skimage import io, img_as_float32, color
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

def apply_median_filter(image, size=3):
    # Calculate the padding width
    pad_size = size // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
    
    # Create sliding window view
    windows = view_as_windows(padded_image, (size, size))
    
    # Compute the median for each window
    output = np.median(windows, axis=(2, 3))
    
    return output

# Load the image and convert it to float32
I = img_as_float32(io.imread('../images/saltpepper.jpg'))  # Load as grayscale

# Convert the image to grayscale if it's not already
if I.ndim == 3:  # Check if the image is RGB
    I_gray = color.rgb2gray(I)  # Convert to grayscale

# Apply the median filter
I_filtered = apply_median_filter(I_gray, size=13)

# Display the images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(I_gray, cmap='gray') 
ax1.set_title('Gray Image')
ax1.axis('off')

ax2.imshow(I_filtered, cmap='gray')
ax2.set_title('Filtered Image')
ax2.axis('off')

plt.show()
