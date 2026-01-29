import cv2
import matplotlib.pyplot as plt
import dlib

print(dlib.__version__)
image = cv2.imread('imagens/px-girl.jpg')


plt.imshow(image)
plt.show()



