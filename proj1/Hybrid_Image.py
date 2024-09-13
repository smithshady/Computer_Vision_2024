import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_low_pass_filter(image, kernel_size=51, sigma=4.0):

    # Split the image into its R, G, and B channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply Gaussian filter to each channel
    b_filtered = cv2.GaussianBlur(b_channel, (kernel_size, kernel_size), sigma)
    g_filtered = cv2.GaussianBlur(g_channel, (kernel_size, kernel_size), sigma)
    r_filtered = cv2.GaussianBlur(r_channel, (kernel_size, kernel_size), sigma)

    # Merge the filtered channels back into an RGB image
    filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))

    return filtered_image

def apply_gaussian_high_pass_filter(image, kernel_size=51, sigma=4.0):

    # Apply Gaussian low-pass filter
    low_pass_filtered = apply_gaussian_low_pass_filter(image, kernel_size, sigma)
    
    # Compute the high-pass filter by subtracting the low-pass filtered image from the original image
    filtered_image = cv2.subtract(image, low_pass_filtered)
    
    return filtered_image

def create_hybrid_image(low_pass_image, high_pass_image, alpha=0.5):

    # Ensure both images are of the same size
    if low_pass_image.shape != high_pass_image.shape:
        raise ValueError(f"Low-pass {low_pass_image.shape} and high-pass {high_pass_image.shape} images must be of the same size")

    # Combine the low-pass and high-pass images
    hybrid_image = cv2.addWeighted(low_pass_image, alpha, high_pass_image, 1 - alpha, 0)
    
    return hybrid_image

# Load the low pass image
image_low = cv2.imread('../images/cat_850_567.jpg')
image_low = cv2.cvtColor(image_low, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format

# Load the high pass image
image_high = cv2.imread('../images/car_1024_768.jpg')
image_high = cv2.cvtColor(image_high, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format

# Apply the Gaussian low-pass filter
filtered_image_low = apply_gaussian_low_pass_filter(image_low)

# Apply the Gaussian low-pass filter
filtered_image_high = apply_gaussian_high_pass_filter(image_high)

hybrid_image = create_hybrid_image(filtered_image_low, filtered_image_high)

# Create a figure and a set of subplots (2 rows x 2 columns)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Plot the images in the subplots
ax1.imshow(image_low)
ax1.set_title('Gray Image')
ax1.axis('off')

ax2.imshow(filtered_image_low)
ax2.set_title('Blurred Image')
ax2.axis('off')

ax3.imshow(image_high)
ax3.set_title('High-Pass Filtered Image')
ax3.axis('off')

ax4.imshow(filtered_image_high)  # Replace 'another_image' with your actual image
ax4.set_title('Another Image')
ax4.axis('off')

# Show the images
plt.tight_layout()
plt.show()

# Create a new figure
plt.figure(figsize=(8, 6))  # You can specify the size of the figure

# Plot the image
plt.imshow(hybrid_image)
plt.title('Hybrid image')
plt.axis('off')  # Hide the axis

# Display the plot
plt.show()