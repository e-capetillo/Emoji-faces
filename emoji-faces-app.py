import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Descargar modelo desde Hugging Face
model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
modelo = YOLO(model_path)

# Cargar imagen
imagen = cv2.imread("tu_imagen.jpg")  # reemplaza con una imagen real
if imagen is None:
    print("Error: Imagen no cargada")
    exit()

# Detectar caras
resultados = modelo(imagen)
caras = resultados[0].boxes

if len(caras) == 0:
    print("No se detectaron caras.")
else:
    for i, box in enumerate(caras):
        coords = box.xyxy[0]
        x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, f"ID {i}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar imagen con caras
    cv2.imshow("Caras detectadas", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
