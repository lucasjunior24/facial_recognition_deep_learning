import cv2
import matplotlib.pyplot as plt
from os import listdir, path, makedirs
from os.path import isfile, join

faces_path = "base/imagens/cropped_faces/"
image_face_1 = cv2.imread(f'{faces_path}s01_01.jpg')
image_face_1 = cv2.cvtColor(image_face_1, cv2.COLOR_BGR2RGB)

image_face_2 = cv2.imread(f'{faces_path}s02_01.jpg')
image_face_2 = cv2.cvtColor(image_face_2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.title("Cara 1")
plt.imshow(image_face_1)

plt.figure(figsize=(20, 10))
plt.subplot(132)
plt.title("Cara 2")
plt.imshow(image_face_2)

# plt.show()

list_filepath_faces = [f for f in listdir(faces_path) if isfile(join(faces_path, f))]
print(list_filepath_faces[0])