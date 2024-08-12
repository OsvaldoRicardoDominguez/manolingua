import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label):
    """
    Carga imágenes desde una carpeta específica y les asigna una etiqueta.

    Parámetros:
    - folder (str): Ruta de la carpeta que contiene subcarpetas con imágenes.
    - label (int): Etiqueta numérica que se asignará a todas las imágenes de esta carpeta.

    Retorna:
    - images (list): Lista de imágenes cargadas y redimensionadas.
    - labels (list): Lista de etiquetas correspondientes a las imágenes.
    """
    images = []
    labels = []
    
    for subdir in os.listdir(folder):  # Recorre cada subcarpeta en la carpeta principal.
        subdir_path = os.path.join(folder, subdir)  # Obtiene la ruta completa de la subcarpeta.
        
        if os.path.isdir(subdir_path):  # Verifica si es un directorio.
            for filename in os.listdir(subdir_path):  # Recorre cada archivo en la subcarpeta.
                img_path = os.path.join(subdir_path, filename)  # Obtiene la ruta completa del archivo de imagen.
                img = cv2.imread(img_path)  # Carga la imagen.
                
                if img is not None:  # Verifica si la imagen se cargó correctamente.
                    img = cv2.resize(img, (224, 224))  # Redimensiona la imagen a 224x224 píxeles.
                    images.append(img)  # Agrega la imagen a la lista.
                    labels.append(label)  # Agrega la etiqueta correspondiente a la lista.
                    
    return images, labels

def load_data(data_dir):
    """
    Carga todas las imágenes y etiquetas de las subcarpetas del directorio principal.

    Parámetros:
    - data_dir (str): Ruta del directorio principal que contiene las carpetas de imágenes.

    Retorna:
    - X (np.ndarray): Array con todas las imágenes cargadas y redimensionadas.
    - y (np.ndarray): Array con todas las etiquetas correspondientes a las imágenes.
    """
    images, labels = [], []
    
    for label, folder in enumerate(['hola', 'adios', 'si']):  # Asigna una etiqueta numérica a cada carpeta.
        img, lbl = load_images_from_folder(os.path.join(data_dir, folder), label)  # Carga imágenes y etiquetas de la carpeta.
        images.extend(img)  # Agrega las imágenes cargadas a la lista principal.
        labels.extend(lbl)  # Agrega las etiquetas correspondientes a la lista principal.
    
    X = np.array(images)  # Convierte la lista de imágenes en un array de numpy.
    y = np.array(labels)  # Convierte la lista de etiquetas en un array de numpy.
    
    return X, y

# Ruta del directorio que contiene las imágenes organizadas en carpetas.
data_dir = 'data'

# Carga las imágenes y sus etiquetas desde el directorio especificado.
X, y = load_data(data_dir)

# Imprime la cantidad total de imágenes cargadas y la distribución de las etiquetas.
print(f"Total images: {len(X)}")
print(f"Labels distribution: {np.bincount(y)}")  # Verifica cuántas imágenes hay por cada etiqueta.

# Divide los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guarda los conjuntos de datos en archivos .npy para su uso posterior.
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
