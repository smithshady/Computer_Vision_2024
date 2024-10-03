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

#find corners using harris
corners = feature.corner_harris(I_gray)
binary_corners = corners > (0.01 * corners.max())  # Adjust threshold as needed
coordinates = np.argwhere(binary_corners)

# Visualize corners on the original image
I_corners = np.copy(I)
I_corners[binary_corners] = [1, 0, 0]  # Mark corners in red
I_corners_uint8 = (I_corners * 255).astype(np.uint8)
show_plot("Detected Harris Corners", I_corners_uint8)


# Count corners detected
# num_corners = np.sum(corners_binary)
# print(f"Number of corners detected: {num_corners}")

#For each corner:
#In a 16x16 window around the corner, compute gradient orientation histogram.
#use a histogram with 36 bins, each covering 10 degrees, to encompass 0 to 360 degrees.
degrees = 10
num_bins = 36
half_size = 8
histogram_arr = []
sub_size = 4
bin_degrees = 360 / 8
last_hist = np.zeros(num_bins)
last_rotated = 0
for y, x in coordinates:
    if x > half_size and y > half_size and x < I_x.shape[1] - half_size and y < I_x.shape[0] - half_size:
        # Extract the 16x16 window around the corner
        window_Ix = I_x[y-half_size:y+half_size, x-half_size:x+half_size]
        window_Iy = I_y[y-half_size:y+half_size, x-half_size:x+half_size]

        window_magnitude = np.sqrt(window_Ix**2 + window_Iy**2)
        window_orientation = np.degrees(np.arctan2(window_Iy, window_Ix)) % 360

        # Create a histogram for this window
        this_hist = np.zeros(num_bins)

        # Loop through each pixel in the 16x16 window and accumulate into histogram
        for i in range(16):
            for j in range(16):
                angle = window_orientation[i, j]
                magnitude = window_magnitude[i, j]
                bin_idx = int(angle // degrees) % num_bins  # Ensure bin index is within bounds
                this_hist[bin_idx] += magnitude  # Accumulate magnitude for that bin

        # Find dominant orientation
        dom_angle = np.argmax(this_hist) * degrees

        # Rotate the orientation values by subtracting the dominant angle
        rotated_orientation = (window_orientation - dom_angle) % 360

        sift_descriptor = []  # Reset for each corner

        for i in range(0, 16, sub_size):
            for j in range(0, 16, sub_size):
                # Extract 4x4 sub-block
                sub_block_orientation = rotated_orientation[i:i + sub_size, j:j + sub_size]
                sub_block_magnitude = window_magnitude[i:i + sub_size, j:j + sub_size]

                # Compute an 8-bin histogram for the sub-block
                sub_hist = np.zeros(8)
                angles_flat = sub_block_orientation.ravel()
                magnitudes_flat = sub_block_magnitude.ravel()

                for idx in range(angles_flat.size):
                    angle = angles_flat[idx]
                    magnitude = magnitudes_flat[idx]
                    sub_bin_idx = int(angle // bin_degrees) % 8  # Ensure sub-bin index is within bounds
                    sub_hist[sub_bin_idx] += magnitude

                # Append the sub-block histogram to the descriptor list
                sift_descriptor.append(sub_hist)

        # Flatten the list of sub-histograms into a single 128-element vector
        sift_descriptor = np.concatenate(sift_descriptor)

        # Normalize the descriptor (to the range 0-1)
        if np.max(sift_descriptor) > 0:
            sift_descriptor = sift_descriptor / np.max(sift_descriptor)

        # Clamp values > 0.2 to 0.2
        sift_descriptor[sift_descriptor > 0.2] = 0.2

        # Re-normalize the descriptor
        if np.sum(sift_descriptor) > 0:
            sift_descriptor = sift_descriptor / np.sum(sift_descriptor)

        # Store the descriptor for this corner
        histogram_arr.append(sift_descriptor)
        last_hist =  this_hist
        last_rotated = window_orientation
