import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Cargar el modelo entrenado para la clasificación de señas
model = tf.keras.models.load_model('modelo_test.h5')

# Configuración de MediaPipe para la detección de manos
# MediaPipe es una biblioteca que facilita la detección y el seguimiento de manos en tiempo real
mp_hands = mp.solutions.hands  # Módulo de MediaPipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils  # Utilidad para dibujar las conexiones entre puntos de la mano

# Configurar el objeto 'hands' con parámetros de confianza para la detección y seguimiento
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Bucle principal para procesar cada cuadro de video en tiempo real
while cap.isOpened():
    ret, frame = cap.read()  # Leer un cuadro de la cámara
    if not ret:  # Verificar si se ha capturado un cuadro correctamente
        break

    # Convertir el cuadro de BGR a RGB, ya que MediaPipe trabaja con imágenes en RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen con MediaPipe para detectar manos
    results = hands.process(frame_rgb)

    # Si se detectan manos en el cuadro, proceder a dibujar y clasificar
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar las conexiones de la mano detectada en el cuadro original
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Obtener las dimensiones del cuadro de video
            h, w, _ = frame.shape
            
            # Calcular las coordenadas mínimas y máximas del cuadro delimitador de la mano
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            
            # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Extraer la región de interés (ROI) de la mano del cuadro original
            hand_img = frame[y_min:y_max, x_min:x_max]
            
            # Redimensionar la imagen de la mano a 224x224 píxeles, que es el tamaño de entrada esperado por el modelo
            hand_img = cv2.resize(hand_img, (224, 224))
            
            # Normalizar los valores de los píxeles de la imagen a un rango de [0, 1]
            hand_img = hand_img / 255.0
            
            # Expandir las dimensiones de la imagen para que sea compatible con el modelo (batch_size, height, width, channels)
            hand_img = np.expand_dims(hand_img, axis=0)
            
            # Realizar la predicción usando el modelo cargado
            prediction = model.predict(hand_img)
            
            # Determinar la clase con la mayor probabilidad
            class_index = np.argmax(prediction)
            
            # Lista de señas conocidas por el modelo
            signs = ['hola', 'adios', 'si']
            
            # Obtener la seña correspondiente al índice de la clase predicha
            sign = signs[class_index]

            # Mostrar la predicción en el cuadro delimitador alrededor de la mano en la imagen
            cv2.putText(frame, sign, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el cuadro de video con las anotaciones
    cv2.imshow('Sign Language Translator', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
