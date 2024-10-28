import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_keypoint(left_img, right_img):
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(left_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(right_img, None)
    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoints(descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2) # Find the two best matches for each descriptor
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance] # Apply Lowe’s ratio test
    return good_matches

if __name__ == "__main__":

    # Load images
    img1 = cv2.imread('../images/HW3/left.jpg')
    img2 = cv2.imread('../images/HW3/right.jpg')
    assert (img1 is not None) and (img2 is not None), 'Cannot read given images'

    # Camera intrinsic matrix
    f, cx, cy = 1000, 1024, 768 
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # Extract and match descriptors
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(img1, img2)
    good_matches = match_keypoints(descriptor1, descriptor2)

    # Extract point coordinates from matches
    pts1 = np.float32([key_points1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([key_points2[m.trainIdx].pt for m in good_matches])

    # Calculate the fundamental matrix using RANSAC
    F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    print(f'* Fundamental Matrix (F) = \n{F}')
    print(f'* Number of inliers = {np.sum(inlier_mask)}')

    # Show matched inlier feature points
    img_matched = cv2.drawMatches(
        img1, key_points1, img2, key_points2, good_matches, None,
        matchesMask=inlier_mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.namedWindow('Fundamental Matrix Estimation', cv2.WINDOW_NORMAL)
    cv2.imshow('Fundamental Matrix Estimation', img_matched)
    cv2.waitKey(0) # PRESS ENTER TO CONTINUE
    cv2.destroyAllWindows()

    # Recover the relative rotation and translation
    E = K.T @ F @ K  # E = K' * F * K
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    print(f'* Essential Matrix (E) = \n{E}')
    print(f'* Rotation Matrix (R) = \n{R}')
    print(f'* Translation Vector (t) = \n{t}')

    # Triangulate points to get 3D coordinates
    P0 = K @ np.eye(3, 4, dtype=np.float32)
    Rt = np.hstack((R, t))  # [R | t]
    P1 = K @ Rt  
    pts1_inlier = pts1[inlier_mask.ravel() == 1]
    pts2_inlier = pts2[inlier_mask.ravel() == 1]
    X = cv2.triangulatePoints(P0, P1, pts1_inlier.T, pts2_inlier.T)
    X /= X[3]  # Convert homogeneous to 3D coordinates
    X = X.T  # Transpose for plotting

    #  Visualize the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'ro', markersize=2)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.grid(True)
    plt.show()
