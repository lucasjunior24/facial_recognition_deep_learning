import pickle

def load_model():
  filename = 'app/model_eingenfaces.pkl'
  with open(filename, 'rb') as file:
      loaded_model = pickle.load(file)
  return loaded_model
print('modelo carregado!')

loaded_model = load_model()
print(loaded_model)

loaded_model.predict()