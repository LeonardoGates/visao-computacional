import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializa MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Definição das casas do violão
CASAS= [
    [696, 121, 64, 99],
    [623, 140, 61, 96],
    [556, 151, 55, 96],
    [492, 160, 54, 105],
    [422, 175, 62, 96],
    [371, 186, 53, 95],
    [298, 195, 74, 97]
]

# Função para detectar as mãos e desenhar os landmarks
def detectar_maos(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return resultados

# Função para verificar se os dedos estão dentro das ROIs
def verificar_dedos_nas_rois(resultados, casas, cordas):
    dedos_nas_casas = []
    dedos_nas_cordas = []
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # Converte a posição normalizada dos landmarks para coordenadas de pixel
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                for i, (x, y, largura, altura) in enumerate(casas):
                    if x < cx < x + largura and y < cy < y + altura:
                        dedos_nas_casas.append(i + 1)  # Adiciona o número da casa (i + 1) à lista
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                for j, (y1, y2) in enumerate(cordas):
                    if min(y1, y2) - 10 < cy < max(y1, y2) + 10:  # Verifica se o dedo está na corda
                        dedos_nas_cordas.append(j + 1)  # Adiciona o número da corda (j + 1) à lista
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    return dedos_nas_casas, dedos_nas_cordas

# Função para exibir as casas e cordas tocadas no canto do vídeo
def exibir_casas_cordas_tocadas(frame, casas_tocadas, cordas_tocadas):
    texto_casas = "Casas tocadas: " + ", ".join(map(str, casas_tocadas))
    texto_cordas = "Cordas tocadas: " + ", ".join(map(str, cordas_tocadas))
    cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame, texto_casas, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, texto_cordas, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Função para detectar as cordas do violão
def detectar_cordas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Aplicar filtro de morfologia para reforçar as linhas horizontais
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Detectar contornos
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cordas = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Ignorar contornos pequenos
            x, y, w, h = cv2.boundingRect(contour)
            if h < 10:  # Considerar apenas contornos horizontais
                cordas.append((y, y + h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cordas

# Captura de vídeo
video_path = 'projeto-final/notas.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cordas = detectar_cordas(frame)
    resultados = detectar_maos(frame)
    casas_tocadas, cordas_tocadas = verificar_dedos_nas_rois(resultados, CASAS, cordas)
    exibir_casas_cordas_tocadas(frame, casas_tocadas, cordas_tocadas)

    for (x, y, largura, altura) in CASAS:
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (255, 0, 0), 2)

    cv2.imshow('Detecção de Mãos, Casas e Cordas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
