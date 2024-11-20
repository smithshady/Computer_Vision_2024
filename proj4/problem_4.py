# NOTE: run this command to install something that will speed up your computing time for the scikit things by a lot on intel devices
%pip install scikit-learn-intelex

#imports
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn import svm
# NOTE: these two lines are running the scikit patch from the import
from sklearnex import patch_sklearn
patch_sklearn()

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]])) 
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)
plt.show()

# Data preprocessing
X_train_orig = np.copy(X_train)
X_test_orig = np.copy(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], -1)) 
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# do more pre-processing as needed
# subtract the mean
X_train_mean = X_train - np.mean(X_train)
# normalize the data
X_train_norm = ((X_train_mean / 255) * 2) - 1
# truncate the data
X_train_processed = X_train_norm[:20000]
y_train_processed = y_train[:20000]

# SVM classifier with linear kernel
clf = svm.SVC(probability=False, kernel='linear', C=0.1)

# fit model
clf.fit(X_train_processed, y_train_processed)

# Evaluate on test set  
predicted = clf.predict(X_test)
score = clf.score(X_test,y_test) #classification score

# print test set score
print("Test set score: ", score)

# Evaluate on training set
train_predicted = clf.predict(X_train)
train_score = clf.score(X_train, y_train)

# print training set score
print("Training set score: ", train_score)

# Test case
i = 10       
xVal = X_test[i, :]
yVal = y_test[i]   
yHat = predicted[i]
xImg = X_test_orig[i]
plt.imshow(xImg)
title = 'true={0:s} est={1:s}'.format(cifar_classes[yVal[0]], cifar_classes[yHat.astype(int)])
plt.title(title)
plt.show()
