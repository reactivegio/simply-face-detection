
import cv2

img_path = input("Insert path of image to analyze: ")
img = cv2.imread(img_path)

# CascadeClassifier works only on grayscaled images so convert your image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Useful staged file of classifier is located on https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# faces returns a tuple list with x, y, w, h 
faces = face_cascade.detectMultiScale(grayImg, 1.1, 2) 
print("Faces found: %d" % len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
