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

# create Sobel filters
x_kernel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
y_kernel = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

# compute image gradients
I_x = convolve(I_gray, x_kernel)
I_y = convolve(I_gray, y_kernel)

# find Hadamard product
dot_prod = np.multiply(I_x, I_y)

# display I_x, I_y, and I_x dot I_y
figure, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(12, 5))
plot1.imshow(I_x, cmap='gray') # use log scaling for magnitude
plot1.set_title('I_x gradient')
plot2.imshow(I_y, cmap='gray')
plot2.set_title('I_y gradient')
plot3.imshow(dot_prod, cmap='gray')
plot3.set_title('Hadamard product')
plt.show()

# compute M componenets as squares of derivatives
I_x2 = I_x ** 2
I_y2 = I_y ** 2

# display
figure, (plot1, plot2) = plt.subplots(1, 2, figsize=(12, 5))
plot1.imshow(I_x2, cmap='gray')
plot1.set_title('I_x squared gradient')
plot2.imshow(I_y2, cmap='gray')
plot2.set_title('I_y squared gradient')
plt.show()

# Gaussian filter
kernel = create_kernel(1.4)

# convolve I_x_sq, I_y_sq, and I_x dot I_y with Gaussian
gaussian_I_x2 = convolve(I_x2, kernel)
gaussian_I_y2 = convolve(I_y2, kernel)
gaussian_hadamard = convolve(dot_prod, kernel)

# display
figure, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(12, 5))
plot1.imshow(gaussian_I_x2, cmap='gray') # use log scaling for magnitude
plot1.set_title('gaussian I_x squared gradient')
plot2.imshow(gaussian_I_y2, cmap='gray')
plot2.set_title('gaussian I_y squared gradient')
plot3.imshow(gaussian_hadamard, cmap='gray')
plot3.set_title('gaussian Hadamard product')
plt.show()

# compute cornerness
alpha = 0.04
C = np.multiply(gaussian_I_x2, gaussian_I_y2) - gaussian_hadamard ** 2 - alpha * ((gaussian_I_x2 + gaussian_I_y2) ** 2)

# display cornerness
show_plot("Cornerness as an image", C)

# create a threshold that of maximum cornerness score
threshold = 0.01 * C.max()
x = I_gray.shape[0]
y = I_gray.shape[1]
threshold_im = np.zeros_like(I_gray)
for i in range(x):
  for j in range(y):
    if C[i][j] >= threshold:
      threshold_im[i][j] = 255

# display threshed C
show_plot("Large corner response: C > threshold", threshold_im)

# use non-maximum suppression (with appropriate threshold) to pick corners as individual pixels
new_threshold = 0.05 * C.max()
non_max_suppression = np.zeros_like(I_gray)
for i in range(6, x - 10):
  for j in range(6, y - 10):
    if C[i][j] >= new_threshold:
      if C[i][j] == C[i-5:i+5, j-5:j+5].max():
        non_max_suppression[i-2:i+2, j-2:j+2] = 255

# display non-maximum suppression
show_plot("Corners as individual pixels", non_max_suppression)

# display corners overlapped on the original image
coordinates = np.where(non_max_suppression > 0)
I_corners = np.copy(I)
for i in range(len(coordinates[0])):
  y, x = coordinates[0][i], coordinates[1][i]
  I_corners[max(y-1,0):min(y+2,I.shape[0]), max(x-1,0):min(x+2,I.shape[1]), :] = [1,0,0]
show_plot("Detected Harris Corners", I_corners)