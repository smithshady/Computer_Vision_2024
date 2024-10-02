import numpy as np
from skimage import io, img_as_float32, color
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

def show_plot(plot_title, plot):
    plt.imshow(plot, cmap='gray')
    plt.title(plot_title)
    plt.axis('off')  # Optionally turn off the axis for a cleaner look
    plt.show()

I = img_as_float32(io.imread('mines.png')).astype(np.float16)

# Remove alpha channel if present
if I.shape[2] == 4:
    I = I[:, :, :3]

# Convert to grayscale if RGB
if I.ndim == 3:
    I_gray = color.rgb2gray(I)

#Compute image gradients IIxx,IIyy via convolving the image with Sobel filters (of appropriate kernel size)
#use them to compute magnitude and orientation of gradient for each pixel.

#For each corner:
#In a 16x16 window around the corner, compute gradient orientation histogram.
#use a histogram with 36 bins, each covering 10 degrees, to encompass 0 to 360 degrees.

#Find the dominant orientation, and normalize orientations by rotating them so that the dominant orientation is in the first bin.

#Create a SIFT descriptor using the (rotated) 16x16 window. That is, use 16 sub-blocks of 4x4 size. 
# For each sub-block, create an 8-bin orientation histogram. 
# Stack the histogram values of all sub-blocks so that a 128-element descriptor vector is created.

#Normalized the descriptor (to the range 0-1). Clamp all vector values > 0.2 to 0.2, and re-normalize.

#For only one of the corners:
#Display the gradient orientation histogram and print the dominant orientation

#Re-compute & display the gradient orientation histogram after rotation
#Display the 8-bin orientation histogram for each sub-block (we have 4x4 sub-blocks, so a total of 16 histograms)

#Print out the 128-element descriptor vector constructed from the histograms
#Print out the normalize descriptor, and re-normalized descriptor
