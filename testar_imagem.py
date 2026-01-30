import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo salvo
modelo = load_model('modelo_lesao.keras')

# Definir as classes na mesma ordem que o treino
classes = ['Estágio 1', 'Estágio 2', 'Estágio 3', 'Estágio 4']

def prever_imagem(caminho_imagem):
    # Carregar e redimensionar a imagem
    img = image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Criar batch

    # Fazer a previsão
    predicao = modelo.predict(img_array)
    indice = np.argmax(predicao)  # Pega o índice da maior probabilidade
    confianca = predicao[0][indice] * 100  # Convertendo em porcentagem

    print(f"Classe prevista: {classes[indice]} ({confianca:.2f}% de confiança)")
    return classes[indice], confianca

# Exemplo de uso
caminho = "teste.jpg"  # Coloque o caminho da imagem que você quer testar
prever_imagem(caminho)
