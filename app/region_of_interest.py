import cv2
import matplotlib.pyplot as plt

image = cv2.imread('imagens/px-people.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image)
# plt.imshow(image)
# plt.show()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier("base/classificadores/haarcascade_frontalface_default.xml")
faces = classifier.detectMultiScale(image_gray, 1.3, 5)
print(len(faces))

face_one = faces[0]
print(face_one)
image_anotation = image.copy()

for (x, y, w, h) in faces:
  cv2.rectangle(image_anotation, (x, y), (x+w, y+h), (255, 255, 0), 2)

plt.figure(figsize=(20, 10))
plt.imshow(image_anotation)
plt.show()