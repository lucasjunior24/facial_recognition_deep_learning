
import cv2
import matplotlib.pyplot as plt
from cv2.typing import MatLike
import cv2.face
from os import listdir
from os.path import isfile, join
import numpy as np
faces_path = "base/imagens/cropped_faces/"

filepath_faces_treino = "base/imagens/treino/"
filepath_faces_teste = "base/imagens/teste/"


def padronizar_imagens(imagem_camimho: str) -> MatLike:
  image: MatLike = cv2.imread(imagem_camimho, cv2.IMREAD_GRAYSCALE)
  image: MatLike  = cv2.resize(image, (200, 200))
  return image

list_faces_treino = [f for f in listdir(filepath_faces_treino) if isfile(join(filepath_faces_treino, f))] 
list_faces_teste = [f for f in listdir(filepath_faces_teste) if isfile(join(filepath_faces_teste, f))] 

def generete_dados(list_faces: list[str], filepath_face: str) -> tuple[list[MatLike], list[str]]:
  dados_treinamento: list[MatLike] = []
  sujeitos_id: list[str] = []

  for arq in list_faces:
    image_path = filepath_face + arq
    image: MatLike = padronizar_imagens(image_path)
    dados_treinamento.append(image)
    sujeito_id = arq[1:3]
    sujeitos_id.append(int(sujeito_id))
  return dados_treinamento, sujeitos_id


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


modelo_LBPH = cv2.face.LBPHFaceRecognizer.create()
modelo_LBPH.train(dados_treinamento, sujeitos)


plt.figure(figsize=(20, 10))

plt.subplot(121)
plt.title("Sujeito " + str(sujeitos_teste[21]))
plt.imshow(dados_teste[21], cmap="gray")

plt.subplot(122)
plt.title("Sujeito " + str(sujeitos_teste[22]))
plt.imshow(dados_teste[22], cmap="gray")

plt.show()

result = modelo_LBPH.predict(dados_teste[22])
print(result)