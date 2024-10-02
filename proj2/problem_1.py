import numpy as np
from skimage import io, img_as_float32, color
import matplotlib.pyplot as plt

def show_plot(plot_title, plot):
    plt.imshow(plot, cmap='gray')
    plt.title(plot_title)
    plt.show()

# Load and preprocess image
I = img_as_float32(io.imread('mines.png'))

# Remove alpha channel if present
if I.shape[2] == 4:
    I = I[:, :, :3]

# Convert to grayscale if RGB
if I.ndim == 3:
    I_gray = color.rgb2gray(I)

# Compute the Fourier transform and shift
I_dft = np.fft.fftshift(np.fft.fft2(I_gray))

# Compute the DFT magnitude and phase
I_mag = np.abs(I_dft)
I_phase = np.angle(I_dft)

# Display magnitude with log scaling
show_plot("DFT Magnitude", (np.log(I_mag + 1)))

# Display phase
show_plot("DFT Phase", I_phase)

flat_mag = I_mag.flatten()
sorted_indices = np.argsort(flat_mag)[::-1] 
top_1000_indices = sorted_indices[:1000] 

# Create a mask 
mask = np.zeros_like(I_mag)
mask_flat = mask.flatten()
mask_flat[top_1000_indices] = 1
mask = mask_flat.reshape(I_mag.shape)

# Apply the mask to the DFT (keep top 1000 DFT coefficients)
I_dft_compressed = I_dft * mask

# Display the compressed DFT magnitude image
I_mag_compressed = np.abs(I_dft_compressed)
show_plot("Compressed DFT Magnitude", (np.log(I_mag_compressed + 1)))

#Reconstruct and display
I_reconstructed =   np.real(np.fft.ifft2(np.fft.ifftshift(I_dft_compressed)))
show_plot("Reconstructed Image", (np.log(I_reconstructed+1)))

#final counts for last question:
original_pixel_count = I_gray.size  # height * width
print(f"Original image has {original_pixel_count} pixels.")

compressed_element_count = 1000
print(f"DFT compression keeps {compressed_element_count} elements.")

compression_ratio = compressed_element_count / original_pixel_count
print(f"Compression ratio is {compression_ratio:.6f}")
