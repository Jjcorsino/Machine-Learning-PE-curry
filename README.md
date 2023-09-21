# Machine-Learning-PE-curry

Modelo de Machine Learning de pose estimation con python para analizar un video de Stephen Curry realizando un lanzamiento.

Dependencias Utilizadas en este repositorio que necesitan ser instaladas mediante !Pip para que funcione

  - CV2
  - Mediapipe
  - Numpy
  - Csv


Se utilizaron landmarks personalizados para tratar de mejorar la precision del modelo. Actualmente se calculan los Angulos de las piernas, el pecho y los brazos del jugador a la hora del lanzamiento para fines analiticos.
El Script da como resultado en archivo CSV con cada frames, angulos de las piernas, pecho y brazos.
Por otro lado tambien se guardaran 3 archivos .mp4 en carpeta raiz llamados pointers, transparent y BlackBox, los cuales contienen el analisis realizado mediante las herramientas anteriores
