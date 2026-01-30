import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- CONFIGURAÇÃO ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25  # mais épocas para melhorar aprendizado

# --- DATA AUGMENTATION PARA TREINO ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    vertical_flip=False
)

train_generator = train_datagen.flow_from_directory(
    "dataset/treino",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# --- VALIDAÇÃO (somente normalização) ---
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    "dataset/validacao",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# --- MODELO BASE ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congelar camadas convolucionais

# --- MODELO FINAL ---
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- CALLBACKS ---
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "modelo_lesao_lpp.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# --- TREINO ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("Treinamento finalizado! Modelo salvo em 'modelo_lesao_lpp.h5'")
