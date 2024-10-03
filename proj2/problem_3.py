# imports
import numpy as np
from skimage import io, img_as_float32, color
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, convolve
import cv2

# load and process the Mines image (again)
im = img_as_float32(io.imread(path))

# remove alpha channel if present
if im.shape[2] == 4:
  im = im[:, :, :3]

# convert to grayscale
im_gray = color.rgb2gray(im)


# create Sobel filters
x_kernel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
y_kernel = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

# perform convolutions to get I_x and I_y
I_x = ndimage.convolve(im_3, x_kernel)
I_y = ndimage.convolve(im_3, y_kernel)

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

# display I_x2, I_y2
figure, (plot1, plot2) = plt.subplots(1, 2, figsize=(12, 5))
plot1.imshow(I_x2, cmap='gray')
plot1.set_title('I_x squared gradient')
plot2.imshow(I_y2, cmap='gray')
plot2.set_title('I_y squared gradient')
plt.show()

# Gaussian filter g() with width s
sigma = 1.4
kernel_size = 2 * int(np.ceil(3 * sigma)) + 1  # Calculate the size of the kernel
kernel = create_kernel(kernel_size, sigma)

# convolve I_x_sq, I_y_sq, and I_x dot I_y with Gaussian
gaussian_I_x2 = ndimage.convolve(I_x2, kernel)
gaussian_I_y2 = ndimage.convolve(I_y2, kernel)
gaussian_hadamard = ndimage.convolve(dot_prod, kernel)

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
plt.imshow(C, cmap='gray')
plt.title("Cornerness as an image")
plt.show()

# create a threshold that of maximum cornerness score
threshold = 0.01 * C.max()
x = im_3.shape[0]
y = im_3.shape[1]
threshold_im = np.zeros_like(im_3)
for i in range(x):
  for j in range(y):
    if C[i][j] >= threshold:
      threshold_im[i][j] = 255

# display threshed C
plt.imshow(threshold_im, cmap='gray')
plt.title("Large corner response: C > threshold")
plt.show()

# use non-maximum suppression (with appropriate threshold) to pick corners as individual pixels
new_threshold = 0.05 * C.max()
non_max_suppression = np.zeros_like(im_3)
for i in range(6, x - 10):
  for j in range(6, y - 10):
    if C[i][j] >= new_threshold:
      if C[i][j] == C[i-5:i+5, j-5:j+5].max():
        non_max_suppression[i-2:i+2, j-2:j+2] = 255

# display non-maximum suppression
plt.imshow(non_max_suppression, cmap='gray')
plt.title('Corners as individual pixels')
plt.show()

# display corners overlapped on the original image
# TODO get corners to overlap original image
overlapped = cv2.addWeighted(im_3, 0.5, non_max_suppression, 0.5, 0)
for i in range(6, x - 10):
  for j in range(6, y - 10):
    if C[i][j] >= new_threshold:
      if C[i][j] == C[i-5:i+5, j-5:j+5].max():
        cv2.rectangle(im_3, (i, j), (i+3, j-3), (255, 0, 0), 2)
plt.imshow(im_3, cmap='gray')
plt.title('corners overlapped on the original image')
plt.show()
