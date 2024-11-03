import cv2
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('csm1.jpg')
img2 = cv2.imread('csm2.jpg')
img3 = cv2.imread('csm3.jpg')
assert ((img1 is not None) and (img2 is not None) and (img3 is not None)), 'Cannot read given images'

images = [img1, img2, img3]

# Load the Haar cascade for face detection
haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process each image
for image in images:
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = haar_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the output image with face detections
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Face Detection')
    plt.axis('off')
    plt.show()
