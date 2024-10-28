# imports
import numpy as np
import cv2
import os
import glob

# define real world coordinates of 3D points using known size of checkerboard pattern
checkerboard_size = (5, 7)

# set criteria for stopping iteration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# vector for 3D points
pt3D = []

# vector for 2D points
pt2D = []

# real world coordinates of 3D points
objpt3D = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objpt3D[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
prev_img_shape = None

# extract path of individual images
images = glob.glob('check*.jpg')

for filename in images:
  # read in images
  image = cv2.imread(filename)
  grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # find chess board corners
  # if desired number of corners are found, ret = true
  ret, corners = cv2.findChessboardCorners(grayColor, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

  # if desired number of corners can be detected them refine pixel coordinates and display them on the images of checkerboard
  if ret == True:
    pt3D.append(objpt3D)

    # refine pixel coordinates for given 2D pts
    corners2 = cv2.cornerSubPix(grayColor, corners, (11,11), (-1,-1), criteria)
    pt2D.append(corners2)

    # draw and display the corners
    image = cv2.drawChessboardCorners(image, checkerboard_size, corners2, ret)

  cv2.imshow(image)
  cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = image.shape[:2]

# perform camera calibration by passing 3D points and corresponding pixel coordinates of the detected corners
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(pt3D, pt2D, grayColor.shape[::-1], None, None)

# display camera matrix:
print(" Camera matrix:") 
print(matrix) 
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs) 
