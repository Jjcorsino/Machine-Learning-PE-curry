import os
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Se importa el video ("")/ webcam (0)
#cap = cv2.VideoCapture(r"triplev2.mp4")
cap = cv2.VideoCapture(0)

column_labels = ["Frame", "Piernas", "Pecho", "Brazo"]

# Da como resultado 3 frames diferentes
output_video = cv2.VideoWriter('Transparent.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
aux_image_video = cv2.VideoWriter('BlackBox.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
pointers_video = cv2.VideoWriter('Pointers.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

base_filename = 'datos_por_frame.csv'
csv_filename = base_filename

# Verificar si el archivo csv ya existe
file_counter = 0
while os.path.exists(csv_filename):
    file_counter += 1
    csv_filename = f'{os.path.splitext(base_filename)[0]}_{file_counter}.csv'

# Crear y escribir en el archivo CSV
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_labels)


# Se inicializa el modelo de pose
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as Holistic:
            manos = []
            for _ in range(1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    
                # Se procesa el frame y se le da tamaño
                    frame = cv2.flip(frame, 2)
                    frame = cv2.resize(frame, (1280, 720))
                    height, width, _ = frame.shape
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = Holistic.process(frame_rgb)
                    hands_results = results.left_hand_landmarks

                # Se dibujan los landmarks
                    if results.pose_landmarks is not None:
                        x0 = int(results.pose_landmarks.landmark[29].x * width)
                        y0 = int(results.pose_landmarks.landmark[29].y * height)
                        x10 = int(results.pose_landmarks.landmark[31].x * width)
                        y10 = int(results.pose_landmarks.landmark[31].y * height)
                        x1 = int(results.pose_landmarks.landmark[27].x * width)
                        y1 = int(results.pose_landmarks.landmark[27].y * height)
                        x2 = int(results.pose_landmarks.landmark[25].x * width)
                        y2 = int(results.pose_landmarks.landmark[25].y * height)
                        x3 = int(results.pose_landmarks.landmark[23].x * width)
                        y3 = int(results.pose_landmarks.landmark[23].y * height)
                        x4 = int(results.pose_landmarks.landmark[11].x * width)
                        y4 = int(results.pose_landmarks.landmark[11].y * height)
                        x5 = int(results.pose_landmarks.landmark[13].x * width)
                        y5 = int(results.pose_landmarks.landmark[13].y * height)
                        x6 = int(results.pose_landmarks.landmark[15].x * width)
                        y6 = int(results.pose_landmarks.landmark[15].y * height)
                        x7 = int(results.pose_landmarks.landmark[1].x * width)
                        y7 = int(results.pose_landmarks.landmark[1].y * height)
                        x8 = int(results.pose_landmarks.landmark[7].x * width)
                        y8 = int(results.pose_landmarks.landmark[7].y * height)

                    # Se calculan los angulos
                        p1 = np.array([x1, y1])
                        p2 = np.array([x2, y2])
                        p3 = np.array([x3, y3])
                        l1 = np.linalg.norm(p2 - p3)
                        l2 = np.linalg.norm(p1 - p3)
                        l3 = np.linalg.norm(p1 - p2)
                        Piernas = degrees(acos((l1**2 + l3**2 - l2**2)/(2*l1*l3)))

                        p4 = np.array([x3, y3])
                        p5 = np.array([x4, y4])
                        p6 = np.array([x5, y5])
                        l4 = np.linalg.norm(p5 - p6)
                        l5 = np.linalg.norm(p4 - p6)
                        l6 = np.linalg.norm(p4 - p5)
                        Pecho = degrees(acos((l4**2 + l6**2 - l5**2)/(2*l4*l6)))

                        p7 = np.array([x4, y4])
                        p8 = np.array([x5, y5])
                        p9 = np.array([x6, y6])
                        l7 = np.linalg.norm(p8 - p9)
                        l8 = np.linalg.norm(p7 - p9)
                        l9 = np.linalg.norm(p7 - p8)
                        Brazos = degrees(acos((l7**2 + l9**2 - l8**2)/(2*l7*l9)))

                    # Se dibujan los angulos
                        aux_image = np.zeros(frame.shape, np.uint8)
                        
                        # Se escriben los datos en el archivo CSV y se trackea el punto 9 de la mano izquierda
                        if hands_results:
                            manos = [hands_results.landmark[9]]
                            points = [Piernas, Pecho, Brazos]
                            for point in points:
                                row = [cap.get(cv2.CAP_PROP_POS_FRAMES), Piernas, Pecho, Brazos]
                                csvwriter.writerow(row)

                        # Calcula la altura del área izquierda del frame y el centro de esta
                        left_area_height = int(height / 2)
                        center_y = int(left_area_height / 2)

                        # Se dibujan las lineas
                        cv2.line(aux_image, (x1, y1), (x2, y2), (0, 255, 96), 4)
                        cv2.line(aux_image, (x2, y2), (x3, y3), (128, 0, 250), 4)
                        cv2.line(aux_image, (x3, y3), (x4, y4), (255, 191, 0), 4)
                        cv2.line(aux_image, (x4, y4), (x5, y5), (0, 255, 255), 4)
                        cv2.line(aux_image, (x5, y5), (x6, y6), (128, 0, 250), 4)
                        cv2.line(aux_image, (x4, y4), (x8, y8), (0, 120, 255), 4)
                        cv2.line(aux_image, (x7, y7), (x8, y8), (0, 191, 0), 4)
                        cv2.line(aux_image, (x0, y0), (x10, y10), (0, 191, 0), 4)
                        cv2.line(aux_image, (x0, y0), (x1, y1), (0, 255, 255), 4)

                        # Se agregan los textos de los angulos 
                        text_piernas = f"Piernas: {int(Piernas)}"
                        text_pecho = f"Pecho: {int(Pecho)}"
                        text_brazos = f"Brazos: {int(Brazos)}"
                        cv2.putText(aux_image, text_brazos, (20, center_y), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(aux_image, text_pecho, (20, center_y + 50), 1, 1.5, (255, 0, 255), 2)
                        cv2.putText(aux_image, text_piernas, (20, center_y + 100), 1, 1.5, (0, 255, 255), 2)


                    # Se dibujan los angulos
                        contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                        contours2 = np.array([[x3, y3], [x4, y4], [x5, y5]])
                        contours3 = np.array([[x4, y4], [x5, y5], [x6, y6]])
                        cv2.fillPoly(aux_image, pts=[contours], color=(0, 255, 255))
                        cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 255))
                        cv2.fillPoly(aux_image, pts=[contours3], color=(0, 255, 0))
                        output = cv2.addWeighted(frame, 1, aux_image, 0.55, 0)

                    # Se dibujan los puntos de las manos
                    for mano in manos:
                        cx, cy = int(mano.x * frame.shape[1]), int(mano.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), 4)
                        cv2.circle(aux_image, (cx, cy), 6, (255, 0, 0), 4)
                        cv2.line(aux_image, (x6, y6), (cx, cy), (255, 0, 0), 4)
                        cv2.line(output, (x6, y6), (cx, cy), (255, 0, 0), 4)
                

                    # Se dibujan los puntos
                        cv2.circle(aux_image, (x1, y1), 5, (0, 255, 255), 4)
                        cv2.circle(aux_image, (x2, y2), 5, (128, 0, 250), 4)
                        cv2.circle(aux_image, (x3, y3), 5, (255, 191, 0), 4)
                        cv2.circle(aux_image, (x4, y4), 5, (255, 25, 235), 4)
                        cv2.circle(aux_image, (x5, y5), 5, (128, 0, 250), 4)
                        cv2.circle(aux_image, (x6, y6), 5, (255, 191, 0), 4)
                        cv2.circle(aux_image, (x7, y7), 5, (0, 191, 0), 4)
                        cv2.circle(aux_image, (x0, y0), 5, (42, 73, 255), 4)
                        cv2.circle(aux_image, (x8, y8), 5, (0, 120, 255), 4)

                    # Se dibujan los puntos en el otro frame
                        cv2.circle(frame, (x1, y1), 5, (0, 255, 255), 4)
                        cv2.circle(frame, (x2, y2), 5, (128, 0, 250), 4)
                        cv2.circle(frame, (x3, y3), 5, (255, 191, 0), 4)
                        cv2.circle(frame, (x4, y4), 5, (255, 25, 235), 4)
                        cv2.circle(frame, (x5, y5), 5, (128, 0, 250), 4)
                        cv2.circle(frame, (x6, y6), 5, (255, 191, 0), 4)
                        cv2.circle(frame, (x7, y7), 5, (0, 191, 0), 4)
                        cv2.circle(frame, (x0, y0), 5, (42, 73, 255), 4)
                        cv2.circle(frame, (x8, y8), 5, (0, 120, 255), 4)


                    # Se muestra el resultado
                        cv2.imshow("Transparent", output)
                        cv2.imshow("BlackBox", aux_image)
                        cv2.imshow("Pointers", frame)

                        output_video.write(output)
                        aux_image_video.write(aux_image)
                        pointers_video.write(frame)

                        if cv2.waitKey(1) & 0xFF == 27:
                            break

    output_video.release()
    aux_image_video.release()
    pointers_video.release()
    cap.release()
    cv2.destroyAllWindows()
  