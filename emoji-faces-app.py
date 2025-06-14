import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import random
import os

# Carga el modelo YOLO para detección de caras
@st.cache_resource
def cargar_modelo():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
    return YOLO(model_path)

modelo = cargar_modelo()

st.title("👾 App de Emojis en Caras - Streamlit")

# Subir imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

# Subir carpeta emojis (no se puede subir carpeta, pero puedes subir emojis individuales o zip)
# Para simplificar, subiremos un zip de emojis y lo extraemos
uploaded_emojis = st.file_uploader("Sube un archivo zip con emojis organizados en carpetas", type=["zip"])

import tempfile
import zipfile

if uploaded_emojis:
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "emojis.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_emojis.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Lista categorías (carpetas)
        categorias = [d for d in os.listdir(tmpdirname) if os.path.isdir(os.path.join(tmpdirname, d))]
        st.sidebar.header("Categorías de emojis")
        categoria = st.sidebar.selectbox("Selecciona categoría", categorias)
        
        emoji_por_categoria = {
            cat: [f for f in os.listdir(os.path.join(tmpdirname, cat)) if f.endswith('.png')]
            for cat in categorias
        }
        
        if uploaded_file and categoria:
            image = Image.open(uploaded_file).convert("RGB")
            imagen_np = np.array(image)[:, :, ::-1]  # RGB a BGR para OpenCV
            
            resultados = modelo(imagen_np)
            caras = resultados[0].boxes
            
            st.write(f"Caras detectadas: {len(caras)}")
            
            # Mostrar imagen con rectángulos y aplicar emojis
            imagen_con_emojis = imagen_np.copy()
            
            # Preparar lista emojis y barajar
            emojis_disponibles = emoji_por_categoria[categoria].copy()
            random.shuffle(emojis_disponibles)
            indice_emoji = 0
            
            for i, box in enumerate(caras):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(imagen_con_emojis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imagen_con_emojis, f"ID {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                
                # Cargar emoji
                emoji_path = os.path.join(tmpdirname, categoria, emojis_disponibles[indice_emoji])
                indice_emoji = (indice_emoji + 1) % len(emojis_disponibles)
                emoji_img = Image.open(emoji_path).convert("RGBA")
                
                ancho = x2 - x1
                alto = y2 - y1
                escala_emoji = int(alto * 1.5)
                emoji_img = emoji_img.resize((escala_emoji, escala_emoji), Image.Resampling.LANCZOS)
                
                # Posición para pegar emoji (centrado horizontal, arriba de la cara)
                pos_x = x1 + (ancho - escala_emoji) // 2
                pos_y = y1 - escala_emoji // 5
                pos_y = max(pos_y, 0)
                
                # Pegar emoji
                img_pil = Image.fromarray(cv2.cvtColor(imagen_con_emojis, cv2.COLOR_BGR2RGB))
                img_pil.paste(emoji_img, (pos_x, pos_y), emoji_img)
                imagen_con_emojis = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            st.image(cv2.cvtColor(imagen_con_emojis, cv2.COLOR_BGR2RGB), caption="Imagen con emojis")
