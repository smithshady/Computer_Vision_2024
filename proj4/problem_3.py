import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

# Data preprocessing
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# Vectorized distance function (Euclidean)
def distance_vectorized(x, data):
    return np.sqrt(np.sum((data - x) ** 2, axis=1))

# kNN function
def kNN(x, k, data, labels):
    distances = distance_vectorized(x, data)
    neighbors = np.argsort(distances)[:k]
    neighbor_labels = labels[neighbors]
    return np.bincount(neighbor_labels).argmax()

# Function to display a test image and prediction
def image_show(i, data, true_label, predicted_label):
    x = data[i].reshape((28, 28))  # Reshape to 28x28 format
    plt.imshow(x, cmap='gray')
    plt.title(f"Predicted={predicted_label}, True={true_label[i]}")
    plt.show()

# Set k value
k = 3

# Test case: Image indexed at 10
i = 10
predicted_class = kNN(X_test[i], k, X_train, y_train)
image_show(i, X_test, y_test, predicted_class)

# Evaluate precision on the first 1000 test samples
correct_predictions = 0
for i in range(1000):
    predicted_class = kNN(X_test[i], k, X_train, y_train)
    if predicted_class == y_test[i]:
        correct_predictions += 1

precision = correct_predictions / 1000
print(f'Precision on first 1000 test samples: {precision:.2f}')

# NOTE: will take a short will to evaluate precision
    
# Explain which distance function you chose for the kNN, and why:

### Euclidean distance was simple to implement and is appropriate in this scenario to measure the straight-line distance between points in the feature space.

# Explain what value of k you used in kNN, and what is the impact of k (i.e., large k vs. small k) 

### I used a value of k=3. Small k leads to more sensitivity to noise. Large k is more robust to nosie but risks over-smoothing.