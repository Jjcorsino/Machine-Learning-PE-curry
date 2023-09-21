# Modelo de Machine Learning para Estimación de Poses con Python
Este repositorio contiene un modelo de Machine Learning desarrollado en Python para analizar un video de Stephen Curry realizando un lanzamiento de baloncesto y estimar sus poses. El modelo utiliza las siguientes dependencias, las cuales deben ser instaladas mediante pip para que el código funcione correctamente:

- CV2: Librería OpenCV para el procesamiento de imágenes y videos.
- Mediapipe: Librería de Google para la detección de landmarks en imágenes y videos.
- Numpy: Librería para cálculos numéricos eficientes.
- Csv: Librería para la manipulación de archivos CSV.


## Características Principales
Este modelo de pose estimation se basa en landmarks personalizados para mejorar la precisión en la estimación de las poses de Stephen Curry al momento de realizar un lanzamiento de baloncesto. El script calcula los ángulos de las piernas, el pecho y los brazos del jugador en cada frame del video con fines analíticos.

## Resultados
El resultado de la ejecución de este script es un archivo CSV que contiene, para cada frame del video, los ángulos de las piernas, el pecho y los brazos del jugador. Además, se generan tres archivos de video en la carpeta raíz con los siguientes nombres:

- pointers.mp4: Este archivo muestra los landmarks y puntos de referencia utilizados para la estimación de poses.
- transparent.mp4: En este archivo se muestra la estimación de poses superpuesta en el video original de Stephen Curry, lo que permite visualizar las poses estimadas de manera transparente.
- BlackBox.mp4: Aquí se presenta el análisis de poses realizado mediante las herramientas mencionadas anteriormente, con un enfoque más detallado y un fondo negro, sin mostrar el video original.

### Instrucciones de Uso

Para clonar este repositorio y colaborar en él, puedes utilizar el siguiente comando de Git:

```
git clone https://github.com/Jjcorsino/Machine-Learning-PE-curry.git
```

Instale las dependencias requeridas utilizando los siguientes comandos:
```
pip install opencv-python
pip install mediapipe
pip install numpy
pip install csv
```

puede ejecutar el script utilizando el siguiente comando

```
python curvyv3.py
```

Después de la ejecución, los resultados se guardarán en un archivo CSV y en los archivos de video mencionados anteriormente en la carpeta raíz del repositorio.
