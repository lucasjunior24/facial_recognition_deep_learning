
import cv2
import matplotlib.pyplot as plt

import cv2.face
from os import listdir
from os.path import isfile, join
import numpy as np
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

def training_model():
  modelo_eingenfaces = cv2.face.EigenFaceRecognizer_create()
  modelo_eingenfaces.train(dados_treinamento, sujeitos)
  return modelo_eingenfaces
  

file_path = 'app/eigenface_model.yml'
def save_model():
  modelo_eingenfaces = cv2.face.EigenFaceRecognizer_create()
  modelo_eingenfaces.train(dados_treinamento, sujeitos)
  print()
  try:
    modelo_eingenfaces.save(file_path)
  except Exception as e:
    print(f"Error pickling object: {e}")



# plt.figure(figsize=(20, 10))

# plt.subplot(121)
# plt.title("Sujeito " + str(sujeitos_teste[6]))
# plt.imshow(dados_teste[6], cmap="gray")

# plt.subplot(122)
# plt.title("Sujeito " + str(sujeitos_teste[7]))
# plt.imshow(dados_teste[7], cmap="gray")

# plt.show()

def load_model():
  recognizer = cv2.face.EigenFaceRecognizer.create()
  recognizer.read(file_path)
  return recognizer


# save_model()l

print('carregar modelo!')

model = load_model()
print('modelo carregado!')

print(dados_teste[6])
result = model.predict(dados_teste[6])
print(result)

result = model.predict(dados_teste[7])
print(result)