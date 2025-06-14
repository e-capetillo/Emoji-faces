import io
import streamlit as st
import cv2
import numpy as np
import os
import random
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Descargar y cargar modelo
model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
modelo = YOLO(model_path)

# Cargar emojis ya incluidos (puedes agregar tus emojis aquí)
EMOJIS_PATH = "emojis"  # carpeta emojis en tu repo

def cargar_emojis():
    categorias = [d for d in os.listdir(EMOJIS_PATH) if os.path.isdir(os.path.join(EMOJIS_PATH, d))]
    emoji_por_categoria = {
        cat: [f for f in os.listdir(os.path.join(EMOJIS_PATH, cat)) if f.lower().endswith('.png')]
        for cat in categorias
    }
    return categorias, emoji_por_categoria

categorias, emoji_por_categoria = cargar_emojis()

st.title("👾 App de Emojis en Caras")

# 1. Carga imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

imagen_bgr = None
caras = []
if uploaded_file:
    imagen_pil = Image.open(uploaded_file).convert("RGB")
    imagen_np = np.array(imagen_pil)
    imagen_bgr = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
    
    # Detectar caras
    resultados = modelo(imagen_bgr)
    caras = resultados[0].boxes
    
    if len(caras) == 0:
        st.warning("No se detectaron caras en la imagen.")
    else:
        # Mostrar imagen con cuadros y IDs (en la imagen)
        imagen_mostrar = imagen_bgr.copy()
        for i, box in enumerate(caras):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(imagen_mostrar, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(imagen_mostrar, f"ID {i}", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        st.image(cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB), caption="Caras detectadas (ID debajo)")

        # Mostrar IDs más grandes y centrados debajo (en columnas)
        cols = st.columns(len(caras))
        for i, col in enumerate(cols):
            col.markdown(f"<h3 style='text-align:center;'>ID: {i}</h3>", unsafe_allow_html=True)

        # Selector múltiple para seleccionar IDs a modificar
        ids_seleccionados = st.multiselect(
            "Selecciona las caras a las que quieres aplicar emojis (IDs):",
            options=list(range(len(caras))),
            default=list(range(len(caras)))
        )

        # Selector de categoría
        categoria = st.selectbox("Selecciona la categoría de emojis:", categorias)
        
        # Botón para aplicar emojis
        if st.button("Aplicar emojis y mostrar preview"):
            imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(imagen_rgb)
            ruta_categoria = os.path.join(EMOJIS_PATH, categoria)
            archivos_emoji = emoji_por_categoria[categoria]
            
            emojis_disponibles = archivos_emoji.copy()
            random.shuffle(emojis_disponibles)
            indice_emoji = 0
            
            for i, box in enumerate(caras):
                if i not in ids_seleccionados:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ancho = x2 - x1
                alto = y2 - y1
                
                if indice_emoji >= len(emojis_disponibles):
                    emojis_disponibles = archivos_emoji.copy()
                    random.shuffle(emojis_disponibles)
                    indice_emoji = 0
                
                emoji_path = os.path.join(ruta_categoria, emojis_disponibles[indice_emoji])
                indice_emoji += 1
                
                emoji_img = Image.open(emoji_path).convert("RGBA")
                escala_emoji = int(alto * 1.5)
                emoji_img = emoji_img.resize((escala_emoji, escala_emoji), Image.Resampling.LANCZOS)
                
                pos_x = x1 + (ancho - escala_emoji) // 2
                pos_y = y1 - escala_emoji // 5
                pos_y = max(pos_y, 0)
                
                imagen_pil.paste(emoji_img, (pos_x, pos_y), emoji_img)
            
            st.image(imagen_pil, caption="Imagen con emojis aplicados")
            
            # Botón para descargar imagen
            buffered = io.BytesIO()
            imagen_pil.save(buffered, format="PNG")
            st.download_button(
                label="Descargar imagen con emojis",
                data=buffered.getvalue(),
                file_name="resultado_emojis.png",
                mime="image/png"
            )
