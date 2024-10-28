import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def stitch(left_img, right_img):
    # extract SIFT keypoints
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    
    # match SIFT descriptors
    good_matches = match_keypoints(descriptor1, descriptor2)
    
    # find homography using ransac
    src_pts = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)# from good_matches
    dst_pts = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # from good_matches
    ransac_reproj_threshold = 5.0  # Threshold in pixels
    confidence = 0.99              # Confidence level
    maxIters = 10000               # Maximum number of iterations for RANSAC
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold, maxIters=maxIters, confidence=confidence)

    # combine images
    rows1, cols1 = left_img.shape[:2]
    rows2, cols2 = right_img.shape[:2]

    points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 =  cv2.perspectiveTransform(points, homography_matrix)
    list_of_points = np.concatenate((points1,points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(homography_matrix)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
    
    # Ensure right_img fits within the output_img bounds
    y_offset, x_offset = -y_min, -x_min
    output_img[y_offset:y_offset+rows2, x_offset:x_offset+cols2] = right_img

    return output_img
    

def get_keypoint(left_img, right_img):
    # find SIFT keypoints
    #initialize the SIFT vector with cv2.SIFT_create()
    sift_vec = cv2.SIFT_create()
    #detect keypoints and compute descriptors for both images
    key_points1, descriptor1 = sift_vec.detectAndCompute(left_img, None)
    key_points2, descriptor2 = sift_vec.detectAndCompute(right_img, None)

    #return the keypoints and descriptors
    return key_points1, descriptor1, key_points2, descriptor2


def match_keypoints(descriptor1, descriptor2):
    # match SIFT descriptors
    # use cv2.BFMatcher() to find knn matches between descriptors of each image pair 
    bfMatcher = cv2.BFMatcher()
    knn_matches = bfMatcher.knnMatch(descriptor1, descriptor2, k=2)
    # appy ratio rest with threshold=.7 to filter matches
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # return matches 
    return good_matches


if __name__ == "__main__":
    # load all 8 images
    img_1 = cv2.imread('field1.jpg')
    img_2 = cv2.imread('field2.jpg')
    img_3 = cv2.imread('field3.jpg')
    img_4 = cv2.imread('field4.jpg')
    img_5 = cv2.imread('field5.jpg')
    img_6 = cv2.imread('field6.jpg')
    img_7 = cv2.imread('field7.jpg')
    img_8 = cv2.imread('field8.jpg')
    assert (img_1 is not None) and (img_2 is not None) and (img_3 is not None) and (img_4 is not None) and (img_5 is not None) and (img_6 is not None) and (img_7 is not None) and (img_8 is not None), 'cannot read given images'

    # downsample images
    height, width = img_1.shape[:2]
    downsize_factor = 10
    img_1 = cv2.resize(img_1, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_2 = cv2.resize(img_2, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_3 = cv2.resize(img_3, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_4 = cv2.resize(img_4, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_5 = cv2.resize(img_5, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_6 = cv2.resize(img_6, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_7 = cv2.resize(img_7, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)
    img_8 = cv2.resize(img_8, (width//downsize_factor, height//downsize_factor), interpolation=cv2.INTER_AREA)

    result_img = stitch(img_8, img_7)
    result_img = stitch(result_img, img_6)
    result_img = stitch(result_img, img_5)
    result_img = stitch(result_img, img_4)
    result_img = stitch(result_img, img_3)
    result_img = stitch(result_img, img_2)
    result_img = stitch(result_img, img_1)

    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Panorama Image')
    plt.axis('off')  # Hide axis
    plt.show()

    cv2.imwrite('panorama.jpg', result_img)
