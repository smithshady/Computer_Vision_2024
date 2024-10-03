import numpy as np
from skimage import io, img_as_float32, color, feature
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


#Compute image gradients Ix,Iy via convolving the image with Sobel filters (of appropriate kernel size)
#use them to compute magnitude and orientation of gradient for each pixel.
dx_filter = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])  # Horizontal Sobel filter

dy_filter = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])  # Vertical Sobel filter

# Compute image gradients via convolving the image with Sobel filters
I_x = convolve(I_gray, dx_filter)  # Gradient in the x-direction
I_y = convolve(I_gray, dy_filter)  # Gradient in the y-direction

# Compute the magnitude of the gradient
magnitude = np.sqrt(I_x**2 + I_y**2)
#orientation
orientation = np.atan2(I_y, I_x)

# Display results
show_plot("Gradient in X Direction (Ix)", I_x)
show_plot("Gradient in Y Direction (Iy)", I_y)
show_plot("Magnitude of Gradient", magnitude)
show_plot("Orientation of Gradient", orientation)

#find corners
corners = feature.corner_harris(I_gray)
binary_corners = corners > (0.01 * corners.max())  # Adjust threshold as needed
coordinates = np.argwhere(binary_corners)

# Visualize corners on the original image (optional)
I_corners = np.copy(I)
I_corners[binary_corners] = [1, 0, 0]  # Mark corners in red
I_corners_uint8 = (I_corners * 255).astype(np.uint8)
show_plot("Detected Harris Corners", I_corners_uint8)

# Count corners detected (optional)
# num_corners = np.sum(corners_binary)
# print(f"Number of corners detected: {num_corners}")

#For each corner: STUCK HERE
#In a 16x16 window around the corner, compute gradient orientation histogram.
#use a histogram with 36 bins, each covering 10 degrees, to encompass 0 to 360 degrees.
for y, x in coordinates:
    # Define the window boundaries
    y_min = max(y - 16 // 2, 0)
    y_max = min(y + 16 // 2, I_gray.shape[0] - 1)
    x_min = max(x - 16 // 2, 0)
    x_max = min(x + 16 // 2, I_gray.shape[1] - 1)

    # Extract the window around the corner
    window_orientation = orientation[y_min:y_max+1, x_min:x_max+1]
    
    # Compute histogram of orientations
    hist, bin_edges = np.histogram(window_orientation, bins=36, range=(-np.pi, np.pi))
    
    # Find the dominant orientation
    dominant_orientation = (np.argmax(hist) * (2 * np.pi / 36)) - np.pi  # Convert to radians

    # Normalize the orientation
    normalized_orientation = (dominant_orientation + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    


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
