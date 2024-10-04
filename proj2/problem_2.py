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

# Create a Gaussian filter kernel with σ = 1.4
kernel = create_kernel(1.4)

# Convolve the kernel with 1D derivate filters
dx_filter = np.array([[1, 0, -1]])
dy_filter = np.array([[1], [0], [-1]])
x_deriv = convolve(kernel, dx_filter)
y_deriv = convolve(kernel, dy_filter)

# Compute image gradients, magnitude, orientation
I_x = convolve(I_gray, x_deriv)
I_y = convolve(I_gray, y_deriv)
magnitude = np.sqrt(I_x**2, I_y**2)
orientation = np.arctan2(I_y, I_x)

# Show result
show_plot("Gaussian Filter Kernel", kernel)
show_plot("X Derivative", x_deriv)
show_plot("Y Derivative", y_deriv)
show_plot("Filtered Image (Ix)", I_x)
show_plot("Filtered Image (Iy)", I_y)
show_plot("Gradient Magnitude", magnitude)
show_plot("Gradient Orientation (Theta)", orientation)

#Remove pixels below a certain threshold
threshold = .08
edges = np.where(magnitude > threshold, magnitude, 0)
show_plot("Edge Detection", edges)