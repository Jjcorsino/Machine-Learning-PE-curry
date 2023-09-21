import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Se importa el video 
cap = cv2.VideoCapture(r"triplev2.mp4")

with open('datos_por_frame.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

# Se inicializa el modelo de pose
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as Holistic:
        for _ in range(5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                
            # Se procesa el frame
                frame = cv2.flip(frame, 2)
                frame = cv2.resize(frame, (700, 500))
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = Holistic.process(frame_rgb)
                hands_results = results.right_hand_landmarks


            # Se dibujan los landmarks
                if results.pose_landmarks is not None:
                    x0 = int(results.pose_landmarks.landmark[30].x * width)
                    y0 = int(results.pose_landmarks.landmark[30].y * height)
                    x10 = int(results.pose_landmarks.landmark[31].x * width)
                    y10 = int(results.pose_landmarks.landmark[31].y * height)
                    x1 = int(results.pose_landmarks.landmark[28].x * width)
                    y1 = int(results.pose_landmarks.landmark[28].y * height)
                    x2 = int(results.pose_landmarks.landmark[26].x * width)
                    y2 = int(results.pose_landmarks.landmark[26].y * height)
                    x3 = int(results.pose_landmarks.landmark[24].x * width)
                    y3 = int(results.pose_landmarks.landmark[24].y * height)
                    x4 = int(results.pose_landmarks.landmark[11].x * width)
                    y4 = int(results.pose_landmarks.landmark[11].y * height)
                    x5 = int(results.pose_landmarks.landmark[13].x * width)
                    y5 = int(results.pose_landmarks.landmark[13].y * height)
                    x6 = int(results.pose_landmarks.landmark[15].x * width)
                    y6 = int(results.pose_landmarks.landmark[15].y * height)
                    x7 = int(results.pose_landmarks.landmark[0].x * width)
                    y7 = int(results.pose_landmarks.landmark[0].y * height)
                    x8 = int(results.pose_landmarks.landmark[7].x * width)
                    y8 = int(results.pose_landmarks.landmark[7].y * height)
                    

                # Se calcula el angulo
                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)
                    angle = degrees(acos((l1**2 + l3**2 - l2**2)/(2*l1*l3)))

                    p4 = np.array([x4, y4])
                    p5 = np.array([x5, y5])
                    p6 = np.array([x6, y6])
                    l4 = np.linalg.norm(p5 - p6)
                    l5 = np.linalg.norm(p4 - p6)
                    l6 = np.linalg.norm(p4 - p5)
                    angle2 = degrees(acos((l4**2 + l6**2 - l5**2)/(2*l4*l6)))



                # Se dibuja el angulo
                    aux_image = np.zeros(frame.shape, np.uint8)
                    if hands_results:
                        points = [hands_results.landmark[20]]
                        for point in points:
                            row = [point.x, point.y]
                            csvwriter.writerow(row)

                    cv2.line(aux_image, (x1, y1), (x2, y2), (0, 255, 255), 4)
                    cv2.line(aux_image, (x2, y2), (x3, y3), (128, 0, 250), 4)
                    cv2.line(aux_image, (x3, y3), (x4, y4), (255, 191, 0), 4)
                    cv2.line(aux_image, (x4, y4), (x5, y5), (0, 255, 255), 4)
                    cv2.line(aux_image, (x5, y5), (x6, y6), (128, 0, 250), 4)
                    cv2.line(aux_image, (x4, y4), (x7, y7), (0, 191, 0), 4)
                    cv2.line(aux_image, (x7, y7), (x8, y8), (0, 191, 0), 4)
                    cv2.line(aux_image, (x0, y0), (x1, y1), (0, 255, 255), 4)
                    cv2.line(aux_image, (x0, y0), (x10, y10), (0, 255, 255), 4)
                    cv2.line(aux_image, (x10, y10), (x1, y1), (0, 255, 255), 4)
                    cv2.putText(aux_image, str(int(angle)), (x2 +40, y2), 1, 1.5, (0, 255, 255), 2)
                    cv2.putText(aux_image, str(int(angle2)), (x4 +40, y4), 1, 1.5, (255, 0, 255), 2)

                # Se dibuja el poligono
                    contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                    contours2 = np.array([[x3, y3], [x4, y4], [x5, y5]])
                    cv2.fillPoly(aux_image, pts=[contours], color=(25, 150, 255))
                    cv2.fillPoly(aux_image, pts=[contours2], color=(250, 50, 25))
                    output = cv2.addWeighted(frame, 1, aux_image, 0.5, 0)
                
                # Dibujar los puntos en el frame
                    for point in points:
                        cx, cy = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), 4)
                        cv2.circle(aux_image, (cx, cy), 6, (255, 0, 0), 4)
                        cv2.line(aux_image, (x6, y6), (cx, cy), (255, 0, 0), 4)
                        cv2.line(output, (x6, y6), (cx, cy), (255, 0, 0), 4)



                        

                # Se dibujan los puntos
                    cv2.circle(aux_image, (x1, y1), 5, (0, 255, 255), 4)
                    cv2.circle(aux_image, (x2, y2), 5, (128, 0, 250), 4)
                    cv2.circle(aux_image, (x3, y3), 5, (255, 191, 0), 4)
                    cv2.circle(aux_image, (x4, y4), 5, (0, 255, 255), 4)
                    cv2.circle(aux_image, (x5, y5), 5, (128, 0, 250), 4)
                    cv2.circle(aux_image, (x6, y6), 5, (255, 191, 0), 4)
                    cv2.circle(aux_image, (x7, y7), 5, (0, 191, 0), 4)
                    cv2.circle(aux_image, (x0, y0), 5, (0, 255, 255), 4)
                    cv2.circle(aux_image, (x10, y10), 5, (0, 255, 255), 4)
                    cv2.circle(aux_image, (x8, y8), 5, (0, 120, 255), 4)



                    cv2.circle(frame, (x1, y1), 5, (0, 255, 255), 4)
                    cv2.circle(frame, (x2, y2), 5, (128, 0, 250), 4)
                    cv2.circle(frame, (x3, y3), 5, (255, 191, 0), 4)
                    cv2.circle(frame, (x4, y4), 5, (0, 255, 255), 4)
                    cv2.circle(frame, (x5, y5), 5, (128, 0, 250), 4)
                    cv2.circle(frame, (x6, y6), 5, (255, 191, 0), 4)
                    cv2.circle(frame, (x7, y7), 5, (0, 191, 0), 4)
                    cv2.circle(frame, (x0, y0), 5, (0, 255, 255), 4)
                    cv2.circle(frame, (x10, y10), 5, (0, 255, 255), 4)
                    cv2.circle(frame, (x8, y8), 5, (0, 120, 255), 4)



                # Se muestra el resultado
                    cv2.imshow("Transparent", output)
                    cv2.imshow("BlackBox", aux_image)
                    cv2.imshow("Pointers", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    cap.release()
    cv2.destroyAllWindows()