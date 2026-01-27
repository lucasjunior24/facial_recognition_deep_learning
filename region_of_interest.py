import cv2
import matplotlib.pyplot as plt

image = cv2.imread('imagens/px-people.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# print(image)
plt.imshow(image)
plt.show()
