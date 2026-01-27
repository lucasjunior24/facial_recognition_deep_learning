import cv2
import matplotlib.pyplot as plt

image = cv2.imread('imagens/px-girl.jpg')

# print(image)
plt.imshow(image)
plt.show()
import dlib
print(dlib.__version__)