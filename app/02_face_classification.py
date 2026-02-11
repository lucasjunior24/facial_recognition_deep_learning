import shutil
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
filepath_faces_treino = "base/imagens/treino"
filepath_faces_teste = "base/imagens/teste"

list_filepath_faces = [f for f in listdir(faces_path) if isfile(join(faces_path, f))]
print(list_filepath_faces[0])

def create_path_treino_and_teste():
  if not path.exists(filepath_faces_treino):
    makedirs(filepath_faces_treino)

  if not path.exists(filepath_faces_teste):
    makedirs(filepath_faces_teste)

def separete_files():
  for arq in list_filepath_faces:
    sujeito = arq[1:3]
    numero = arq[4:6]
    if int(numero) <= 10:
      shutil.copyfile(faces_path + arq, filepath_faces_treino + arq)
    else:
      shutil.copyfile(faces_path + arq, filepath_faces_teste + arq)

def padronizar_imagens(imagem_camimho: str):
  image = cv2.imread(imagem_camimho, cv2.IMREAD_GRAYSCALE)
  print(image)
  image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LANCZOS4)
  return image

list_faces_treino = [f for f in listdir(filepath_faces_treino) if isfile(join(filepath_faces_treino, f))] 
list_faces_teste = [f for f in listdir(filepath_faces_teste) if isfile(join(filepath_faces_teste, f))] 
print()
print("teste treino: ", list_faces_treino[0])
print("teste file: ", list_faces_teste[0])
print(len(list_faces_treino))
print()
print('generete_dados')
print()
def generete_dados():
  dados_treinamento, sujeitos = [], []
  for i, arq in enumerate(list_faces_treino):
    image_path = filepath_faces_treino + arq
    image = padronizar_imagens(image_path)
    dados_treinamento.append(image)
    sujeito = arq[1:3]
    sujeitos.append(int(sujeito))
  return dados_treinamento, sujeitos

print()
dados_treinamento, sujeitos = generete_dados()

print(len(dados_treinamento))
print(len(sujeitos))