import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminhos dos datasets
treino_dir = "dataset/treino"
validacao_dir = "dataset/validacao"

# Data Augmentation para treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       # Rotação aleatória
    width_shift_range=0.1,   # Translação horizontal
    height_shift_range=0.1,  # Translação vertical
    shear_range=0.1,         # Corte angular
    zoom_range=0.2,          # Zoom aleatório
    horizontal_flip=True,    # Flip horizontal
    brightness_range=[0.7,1.3], # Alteração de brilho
    fill_mode='nearest'
)

# Apenas normalização para validação
val_datagen = ImageDataGenerator(rescale=1./255)

# Preparando os datasets
train_generator = train_datagen.flow_from_directory(
    treino_dir,
    target_size=(150,150),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validacao_dir,
    target_size=(150,150),
    batch_size=16,
    class_mode='categorical'
)

# Construção do modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout para evitar overfitting
    Dense(4, activation='softmax')  # 4 classes de estágio
])

# Compilando o modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=5,  # Para caso não melhore mais
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'modelo_lesao.keras', 
    monitor='val_accuracy',
    save_best_only=True
)

# Treinamento
history = model.fit(
    train_generator,
    epochs=30,  # Mais epochs, EarlyStopping interrompe se não melhorar
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

print("Treinamento finalizado e modelo salvo em 'modelo_lesao.keras'!")
