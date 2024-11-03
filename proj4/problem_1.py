import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('csm1.jpg')
img2 = cv2.imread('csm2.jpg')
img3 = cv2.imread('csm3.jpg')
assert ((img1 is not None) and (img2 is not None) and (img3 is not None)), 'Cannot read given images'

plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Input Image 1')
plt.axis('off')
plt.show()

plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Input Image 2')
plt.axis('off')
plt.show()

plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title('Input Image 3')
plt.axis('off')
plt.show()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

images = [img1, img2, img3]

for image in images:

    regions, weights = hog.detectMultiScale(image)  
   
    regions = np.array([[x, y, x + w, y + h] for (x, y, w, h) in regions])
    weights = weights.flatten()  

    # Non-maxima supression
    if len(regions) > 0:
        picks = cv2.dnn.NMSBoxes(regions.tolist(), weights.tolist(), score_threshold=0.5, nms_threshold=0.4)
    else:
        picks = []

    # Draw boxes
    if len(picks) > 0:
        for i in picks.flatten():  
            (xA, yA, xB, yB) = regions[i]
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    # Display results
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Result')
    plt.axis('off')
    plt.show()
