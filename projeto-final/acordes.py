import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

CASAS = [
    [696, 121, 64, 99],
    [623, 140, 61, 96],
    [556, 151, 55, 96],
    [492, 160, 54, 105],
    [422, 175, 62, 96],
    [371, 186, 53, 95],
    [298, 195, 74, 97]
]


MAPPING_NOTAS = {
    (1, 6): "F",  (2, 6): "F#",  (3, 6): "G",  (4, 6): "G#",  (5, 6): "A",  (6, 6): "A#",  (7, 6): "B",
    (1, 5): "A#",  (2, 5): "B",  (3, 5): "C",  (4, 5): "C#",  (5, 5): "D",  (6, 5): "D#",  (7, 5): "E",
    (1, 4): "D#",  (2, 4): "E",  (3, 4): "F",  (4, 4): "F#",  (5, 4): "G",  (6, 4): "G#",  (7, 4): "A",
    (1, 3): "G#",  (2, 3): "A",  (3, 3): "A#",  (4, 3): "B",  (5, 3): "C",  (6, 3): "C#",  (7, 3): "D",
    (1, 2): "C",  (2, 2): "C#",  (3, 2): "D",  (4, 2): "D#",  (5, 2): "E",  (6, 2): "F",  (7, 2): "F#",
    (1, 1): "F",  (2, 1): "F#",  (3, 1): "G",  (4, 1): "G#",  (5, 1): "A",  (6, 1): "A#",  (7, 1): "B",
}


buffer_tamanho = 10
historico_dedos = []


def detectar_maos(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return resultados


def verificar_dedo_indicador_nas_rois(resultados, casas, cordas):
    global historico_dedos
    dedo_nas_casas = []
    notas = []
    finger_tip_id = 8 
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            lm = hand_landmarks.landmark[finger_tip_id]
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            casa_tocada = None
            corda_tocada = None

           
            for i, (x, y, largura, altura) in enumerate(casas):
                if x < cx < x + largura and y < cy < y + altura:
                    casa_tocada = i + 1
                    break  

           
            for j, (y1, y2) in enumerate(cordas):
                if min(y1, y2) < cy < max(y1, y2): 
                    corda_tocada = j + 1
                    break  

            if casa_tocada and corda_tocada:
                dedo_nas_casas.append(casa_tocada)
                nota = MAPPING_NOTAS.get((casa_tocada, corda_tocada), "")
                if nota:
                    notas.append(nota)

        
            if len(historico_dedos) >= buffer_tamanho:
                historico_dedos.pop(0)
            historico_dedos.append((casa_tocada, corda_tocada))

    
    if historico_dedos:
        casas_tocadas = [x[0] for x in historico_dedos if x[0] is not None]
        cordas_tocadas = [x[1] for x in historico_dedos if x[1] is not None]
        casa_tocada = int(np.median(casas_tocadas)) if casas_tocadas else None
        corda_tocada = int(np.median(cordas_tocadas)) if cordas_tocadas else None
        if casa_tocada and corda_tocada:
            nota = MAPPING_NOTAS.get((casa_tocada, corda_tocada), "")
            return [casa_tocada], [nota]
    return dedo_nas_casas, notas

def exibir_casas_notas_tocadas(frame, casas_tocadas, notas):
    texto_casas = "Casas tocadas: " + ", ".join(map(str, casas_tocadas))
    texto_notas = "Notas: " + ", ".join(notas)
    cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
    cv2.putText(frame, texto_casas, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, texto_notas, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def detectar_cordas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  

    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    cordas = []
    
    if lines is not None:
        horizontais = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 30:  
                horizontais.append((y1, y2))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if horizontais:
            horizontais = sorted(horizontais, key=lambda y: min(y[0], y[1]), reverse=True)
            cordas = [horizontais[0]]
            for i in range(1, len(horizontais)):
                prev_y_avg = (cordas[-1][0] + cordas[-1][1]) / 2
                current_y_avg = (horizontais[i][0] + horizontais[i][1]) / 2
                if abs(current_y_avg - prev_y_avg) > 20: 
                    cordas.append(horizontais[i])
            while len(cordas) < 6:
                cordas.append(cordas[-1])  
            cordas = cordas[:6]
    
    return cordas

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
    casas_tocadas, notas_tocadas = verificar_dedo_indicador_nas_rois(resultados, CASAS, cordas)
    exibir_casas_notas_tocadas(frame, casas_tocadas, notas_tocadas)

    for (x, y, largura, altura) in CASAS:
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (255, 0, 0), 2)

    cv2.imshow('Detecção de Mãos, Casas e Notas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
