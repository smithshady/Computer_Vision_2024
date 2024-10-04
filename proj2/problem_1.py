# Imports
import numpy as np
from skimage import io, img_as_float32, color, feature
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
import cv2

# plot function
def show_plot(plot_title, plot):
  plt.imshow(plot, cmap='gray')
  plt.title(plot_title)
  plt.show()

# gaussian kernel function
def create_kernel(sigma):
  size = 2 * int(np.ceil(3 * sigma)) + 1
  kernel = np.zeros((size, size))
  kernel[size//2, size//2] = 1
  return gaussian_filter(kernel, sigma)

# load image function
def load_image(image):
  I = img_as_float32(io.imread(image))
  # shape to RGB
  if I.shape[2] == 4:
    I = I[:, :, :3]
  # return original image and gray image
  return I, color.rgb2gray(I)

I, I_gray = load_image('mines.png')

# Compute the Fourier transform, DFT mag, phase
I_dft = np.fft.fftshift(np.fft.fft2(I_gray))
I_mag = np.abs(I_dft)
I_phase = np.angle(I_dft)

# Create a mask
compressed_element_count = 1000
sorted_indices = np.argsort(I_mag.flatten())[::-1]
top_1000_indices = sorted_indices[:compressed_element_count]
mask = np.zeros_like(I_mag).flatten()
mask[top_1000_indices] = 1
mask = mask.reshape(I_mag.shape)

# Compress, reconstruct
I_dft_compressed = I_dft * mask
I_mag_compressed = np.abs(I_dft_compressed)
I_reconstructed =   np.real(np.fft.ifft2(np.fft.ifftshift(I_dft_compressed)))

# Show results
show_plot("DFT Magnitude", (np.log(I_mag + 1)))
show_plot("DFT Phase", I_phase)
show_plot("Compressed DFT Magnitude", (np.log(I_mag_compressed + 1)))
show_plot("Reconstructed Image", (np.log(I_reconstructed+1)))

# Answer questions
original_pixel_count = I_gray.size  # height * width
compression_ratio = compressed_element_count / original_pixel_count
print(f"Original image has {original_pixel_count} pixels.")
print(f"DFT compression keeps {compressed_element_count} elements.")
print(f"Compression ratio is {compression_ratio:.6f}")