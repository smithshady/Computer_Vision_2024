import numpy as np
from scipy.signal import correlate2d
from skimage import io, img_as_float32, color
import matplotlib.pyplot as plt

# Load the images and convert them to float32
I = img_as_float32(io.imread('../images/BlasterID.jpg'))
k = img_as_float32(io.imread('../images/BlasterIDeye2.jpg'))

# Convert the images to grayscale if they're in RGB
if I.ndim == 3:
    I_gray = color.rgb2gray(I) 
if k.ndim == 3:
    k = color.rgb2gray(k)

# Print dimensions
print(f'image shape: {I.shape} | kernel shape: {k.shape}')

# Subtracting the mean so zero centered
I_gray2 = I_gray - np.mean(I)
k2 = k - np.mean(k)

# Perform correlation
I_correlate = correlate2d(I_gray2, k2, 'same')

# Create a figure and a set of subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

ax1.imshow(I_gray, cmap='gray') 
ax1.set_title('Gray Image')
ax1.axis('off')

ax2.imshow(k, cmap='gray') 
ax2.set_title('Eye Template')
ax2.axis('off')

ax3.imshow(I_correlate, cmap='gray') 
ax3.set_title('Correlation')
ax3.axis('off')

# Show the images
plt.tight_layout()
plt.show()

