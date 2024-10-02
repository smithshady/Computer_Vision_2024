import numpy as np
from skimage import io, img_as_float32, color
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

def show_plot(plot_title, plot):
    plt.imshow(plot, cmap='gray')
    plt.title(plot_title)
    plt.axis('off')  # Optionally turn off the axis for a cleaner look
    plt.show()

def create_kernel(size, sigma):
    kernel = np.zeros((size, size))
    kernel[size//2, size//2] = 1
    return gaussian_filter(kernel, sigma)

# Load and preprocess image
I = img_as_float32(io.imread('mines.png')).astype(np.float16)

# Remove alpha channel if present
if I.shape[2] == 4:
    I = I[:, :, :3]

# Convert to grayscale if RGB
if I.ndim == 3:
    I_gray = color.rgb2gray(I)

# Create a Gaussian filter kernel with σ = 1.4 and display
sigma = 1.4
kernel_size = 2 * int(np.ceil(3 * sigma)) + 1  # Calculate the size of the kernel
kernel = create_kernel(kernel_size, sigma)
show_plot("Gaussian Filter Kernel", kernel)


#Convolve the Gaussian kernel with 1D derivate filter kernels along the x and y directions to obtain derivative of Gaussian kernels

# Define 1D derivative filters for x and y directions
dx_filter = np.array([1, 0, -1]).reshape(1, -1)  # Horizontal Sobel filter
dy_filter = np.array([[1], [0], [-1]])  # Vertical Sobel filter

# Convolve the kernel with the derivate filters
x_deriv = convolve(kernel, dx_filter)  # Derivative in x direction
y_deriv = convolve(kernel, dy_filter)  # Derivative in y direction

# Show results
show_plot("X Derivative", x_deriv)
show_plot("Y Derivative", y_deriv)

# Filter the Mines image with derivative of Gaussian filters to obtain Ix and Iy
I_x = convolve(I_gray, x_deriv)  # Convolve with x-derivative of Gaussian
I_y = convolve(I_gray, y_deriv)  # Convolve with y-derivative of Gaussian

# Display IIxx and IIyy
show_plot("Filtered Image (Ix)", I_x)  # Display horizontal gradient
show_plot("Filtered Image (Iy)", I_y)  # Display vertical gradient

#Compute the magnitude and orientation (theta) of gradient
#Display both magnitude and orientation of gradient as images
magnitude = np.sqrt(I_x**2, I_y**2)
orientation = np.atan2(I_y, I_x)

show_plot("Gradient Magnitude", magnitude)
show_plot("Gradient Orientation (Theta)", orientation)

#Remove pixels in the magnitude image that are below a certain threshold (pick the threshold appropriately to keep edges)
#Display the resulting edge image
threshold = .1
edges = np.where(magnitude > threshold, magnitude, 0)
show_plot("Edge Detection", edges)
