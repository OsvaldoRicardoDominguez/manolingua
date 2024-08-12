import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Carga de los datos de entrenamiento y prueba previamente guardados en archivos .npy
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Realiza el One-hot encoding de las etiquetas para convertirlas en vectores binarios.
# Esto es necesario porque las redes neuronales, incluyendo las CNN, trabajan mejor con representaciones categóricas.
y_train = to_categorical(y_train, num_classes=3)  # 3 clases: hola, adios, si
y_test = to_categorical(y_test, num_classes=3)

# Definir la arquitectura de la Red Neuronal Convolucional (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Primera capa de convolución con 32 filtros
    MaxPooling2D((2, 2)),  # Primera capa de max-pooling para reducir la dimensión espacial
    Conv2D(64, (3, 3), activation='relu'),  # Segunda capa de convolución con 64 filtros
    MaxPooling2D((2, 2)),  # Segunda capa de max-pooling
    Conv2D(128, (3, 3), activation='relu'),  # Tercera capa de convolución con 128 filtros
    MaxPooling2D((2, 2)),  # Tercera capa de max-pooling
    Flatten(),  # Aplana la salida para conectarla a la capa densa
    Dense(128, activation='relu'),  # Capa completamente conectada con 128 neuronas
    Dropout(0.5),  # Capa de Dropout para evitar el sobreajuste, desconecta el 50% de las neuronas
    Dense(3, activation='softmax')  # Capa de salida con 3 neuronas para las 3 clases, con activación softmax
])

# Compila el modelo definiendo el optimizador, la función de pérdida y la métrica de evaluación
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo con los datos de entrenamiento y validación con 15 épocas
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Guarda el modelo entrenado en un archivo .h5 para uso posterior
model.save('modelo_test.h5')
