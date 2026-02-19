
import cv2
import matplotlib.pyplot as plt

import cv2.face
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import accuracy_score
faces_path = "base/imagens/cropped_faces/"

filepath_faces_treino = "base/imagens/treino/"
filepath_faces_teste = "base/imagens/teste/"


def padronizar_imagens(imagem_camimho: str):
  image = cv2.imread(imagem_camimho, cv2.IMREAD_GRAYSCALE)
  image = cv2.resize(image, (200, 200))
  return image

list_faces_treino = [f for f in listdir(filepath_faces_treino) if isfile(join(filepath_faces_treino, f))] 
list_faces_teste = [f for f in listdir(filepath_faces_teste) if isfile(join(filepath_faces_teste, f))] 

def generete_dados(list_faces: list[str], filepath_face: str):
  dados_treinamento, sujeitos = [], []
  for i, arq in enumerate(list_faces):
    image_path = filepath_face + arq
    image = padronizar_imagens(image_path)
    dados_treinamento.append(image)
    sujeito = arq[1:3]
    sujeitos.append(int(sujeito))
  return dados_treinamento, sujeitos


print()
print("TREINO")
dados_treinamento, sujeitos_treino = generete_dados(list_faces_treino, filepath_faces_treino)

print(len(dados_treinamento))
print(len(sujeitos_treino))

print()
print("TESTE")
dados_teste, sujeitos_teste = generete_dados(list_faces_teste, filepath_faces_teste)

print(len(dados_teste))
print(len(sujeitos_teste))

sujeitos = np.asarray(sujeitos_treino, dtype=np.int32)
sujeitos_teste = np.asarray(sujeitos_teste, dtype=np.int32)


modelo_fisherfaces = cv2.face.FisherFaceRecognizer.create()
modelo_fisherfaces.train(dados_treinamento, sujeitos)


# plt.figure(figsize=(20, 10))

# plt.subplot(121)
# plt.title("Sujeito " + str(sujeitos_teste[13]))
# plt.imshow(dados_teste[13], cmap="gray")

# plt.subplot(122)
# plt.title("Sujeito " + str(sujeitos_teste[19]))
# plt.imshow(dados_teste[19], cmap="gray")

# plt.show()

result = modelo_fisherfaces.predict(dados_teste[13])
print(result)

y_pred = []

for dado in dados_teste:
  y_pred.append(modelo_fisherfaces.predict(dado)[0])

accuracy = accuracy_score(sujeitos_teste, y_pred)
print('accuracy: ')
print(accuracy)