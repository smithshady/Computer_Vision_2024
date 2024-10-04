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

# Find corners using harris detector
def find_corners(image, image_gray, threshold):
  corners = feature.corner_harris(image_gray)
  binary_corners = corners > (threshold * corners.max())
  coordinates = np.argwhere(binary_corners)
  print(f"Number of corners detected: {len(coordinates)}")
  # Visualize corners on the original image
  I_corners = np.copy(image)
  for coord in coordinates:
      cv2.circle(I_corners, (coord[1], coord[0]), 10, (1, 0, 0), -1)
  I_corners_uint8 = (I_corners * 255).astype(np.uint8)
  show_plot("Detected Harris Corners", I_corners_uint8)
  return coordinates

# SIFT function
def SIFT(I_x, I_y, coordinates, print_corner):
  degrees = 10
  num_bins = 36
  half_size = 8
  histogram_arr = []
  sub_size = 4
  bin_degrees = 360 / 8
  last_hist = np.zeros(num_bins)
  last_rotated = 0
  # Loop through each corner
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

      # Print
      if print_corner:
        print(f"Dominant Orientation: {dom_angle} degrees")
        # Display the gradient orientation histogram before rotation
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(num_bins) * degrees, this_hist, width=degrees, color='b', alpha=0.7)
        plt.title('Gradient Orientation Histogram (Before Rotation)')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 360)
        plt.show()

      # Rotate the orientation values by subtracting the dominant angle
      rotated_orientation = (window_orientation - dom_angle) % 360

      sift_descriptor = []  # Reset for each corner

      # Print
      if print_corner:
        sub_histograms = []  # For visualizing each 4x4 sub-block histogram

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

          # Print
          if print_corner:
            sub_histograms.append(sub_hist)

      # Print
      if print_corner:
        # Re-compute and display the gradient orientation histogram after rotation
        rotated_hist = np.zeros(8)
        for idx in range(rotated_orientation.ravel().size):
            angle = rotated_orientation.ravel()[idx]
            magnitude = window_magnitude.ravel()[idx]
            bin_idx = int(angle // bin_degrees) % 8
            rotated_hist[bin_idx] += magnitude

        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(8) * bin_degrees, rotated_hist, width=bin_degrees, color='r', alpha=0.7)
        plt.title('Gradient Orientation Histogram (After Rotation)')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 360)
        plt.show()

        # Display the 8-bin histogram for each 4x4 sub-block (total 16 histograms)
        for idx, sub_hist in enumerate(sub_histograms):
            plt.figure(figsize=(4, 2))
            plt.bar(np.arange(8) * bin_degrees, sub_hist, width=bin_degrees, color='g', alpha=0.7)
            plt.title(f'Sub-block {idx + 1} Orientation Histogram')
            plt.xlabel('Orientation (degrees)')
            plt.ylabel('Magnitude')
            plt.xlim(0, 360)
            plt.show()

      # Flatten the list of sub-histograms into a single 128-element vector
      sift_descriptor = np.concatenate(sift_descriptor)

      # Print
      if print_corner:
        print("128-element SIFT descriptor:")
        print(sift_descriptor)

      # Normalize the descriptor (to the range 0-1)
      if np.max(sift_descriptor) > 0:
          sift_descriptor = sift_descriptor / np.max(sift_descriptor)

      # Print
      if print_corner:
        print("Normalized SIFT descriptor:")
        print(sift_descriptor)

      # Clamp and re-normalize the descriptor
      sift_descriptor[sift_descriptor > 0.2] = 0.2
      if np.sum(sift_descriptor) > 0:
          sift_descriptor = sift_descriptor / np.sum(sift_descriptor)

      # Print
      if print_corner:
        print("Re-normalized SIFT descriptor:")
        print(sift_descriptor)
        print_corner = False

      # Store the descriptor for this corner
      histogram_arr.append(sift_descriptor)
      last_hist =  this_hist
      last_rotated = window_orientation

  # Return all feature descriptors
  return histogram_arr

# Run function (for problem 5 convenience)
def run_SIFT(image, corner_threshold, print_corner):
  # load images
  I, I_gray = load_image(image)
  # create Sobel filters
  x_kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
  y_kernel = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])
  # Compute image gradients
  I_x = convolve(I_gray, x_kernel)
  I_y = convolve(I_gray, y_kernel)
  # Get corners
  coordinates = find_corners(I, I_gray, corner_threshold)
  # Run the SIFT algorithm
  h = SIFT(I_x, I_y, coordinates, print_corner)
  return I, coordinates, h

# Run SIFT
I, c, h = run_SIFT('image1.jpg', 0.15, True)