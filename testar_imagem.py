import tensorflow as tf
import numpy as np
import cv2

# Carregar o modelo treinado
model = tf.keras.models.load_model("modelo_lesao_lpp.h5")

# Definir as classes
classes = ["Estágio 1", "Estágio 2", "Estágio 3", "Estágio 4"]

# Ler a imagem
img = cv2.imread("teste.jpg")  # coloque o caminho correto se for diferente
if img is None:
    raise FileNotFoundError("Imagem não encontrada. Verifique o nome e a pasta!")

# Pré-processar a imagem igual ao treino
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Fazer a previsão
pred = model.predict(img)[0]  # pegar o array de probabilidades

# Mostrar as probabilidades de cada estágio
print("Probabilidades por estágio:")
for i, p in enumerate(pred):
    print(f"{classes[i]}: {p*100:.2f}%")

# Mostrar a classe final prevista
print("\nClasse prevista:", classes[np.argmax(pred)])
