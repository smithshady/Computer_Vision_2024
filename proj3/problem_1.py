# imports
import math
import numpy as np

# What is the point representation in homogeneous coordinates?
p = [[10],
     [20],
     [1]]
print(p)

# What is the rotation matrix R?
x = 10
y = 20
# convert to radians
theta = 45 * math.pi / 180
R = [[math.cos(theta), -math.sin(theta), 0],
     [math.sin(theta), math.cos(theta), 0],
     [0, 0, 1]]
print(R)

# What is the translation vector t?
t = [[40],
     [-30],
     [1]]
print(t)

# What is the full transformation matrix (consisting of R,t) that can be used to transform the homogeneous point coordinate?
R_t = [[math.cos(theta), -math.sin(theta), 40],
     [math.sin(theta), math.cos(theta), -30],
     [0, 0, 1]]
print(R_t)

# How do we apply this transformation to the point (in homogeneous coordinate form)?
# matrix multiplication!!
transformed_p = np.matmul(R_t, p)

# What is the coordinate of the transformed point, in homogeneous coordinates,
print(transformed_p)

# and in the cartesian coordinates?
print("(", transformed_p[0][0], ", ", transformed_p[1][0], ")")

# 3D transformation
# What is cHw? Explain how you computed it
# The homogeneous transformation from the origin of the world frame to the origin of the camera is:
t = [[0],
     [-5],
     [3],
     [1]]
# The rotation matrix about the x-axis follows the following format:
theta = -30 * math.pi / 180
R = [[1, 0, 0],
     [0, math.cos(theta), -math.sin(theta)],
     [0, math.sin(theta), math.cos(theta)]]

# put the two together in the form of [R | t] for cHw:
cHw = [[1, 0, 0, 0],
       [0, math.cos(theta), -math.sin(theta), -5],
       [0, math.sin(theta), math.cos(theta), 3],
       [0, 0, 0, 1]]
print(cHw)

# using cHw, transform w_p in the world frame to the camera frame
w_p = [[0],
       [0],
       [1],
       [1]]

w_p_transformed = np.matmul(cHw, w_p)
print(w_p_transformed)

# transformed w_p in cartesian coordinates:
print("(", w_p_transformed[0][0], ", ", w_p_transformed[1][0], ", ", w_p_transformed[2][0], ")")
