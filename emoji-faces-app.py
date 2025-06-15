import io
import streamlit as st
import cv2
import numpy as np
import os
import random
import zipfile
import tempfile
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Descargar y cargar modelo YOLO
model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
modelo = YOLO(model_path)

# Función para cargar emojis desde una ruta
def cargar_emojis(desde_path):
    categorias = [d for d in os.listdir(desde_path) if os.path.isdir(os.path.join(desde_path, d))]
    emoji_por_categoria = {
        cat: [f for f in os.listdir(os.path.join(desde_path, cat)) if f.lower().endswith('.png')]
        for cat in categorias
    }
    return categorias, emoji_por_categoria

# Título de la app
st.title("👾 App de Emojis en Caras")

# Opción para subir ZIP con emojis personalizados
st.markdown("### 📦 Subir emojis personalizados (opcional)")
emoji_zip = st.file_uploader("Sube un archivo ZIP con tus emojis (estructura: categoría/cara1.png...)", type=["zip"])

# Definir ruta de emojis: temporal o del repo
if emoji_zip:
    directorio_temporal = tempfile.TemporaryDirectory()
    ruta_emojis = directorio_temporal.name
    with zipfile.ZipFile(emoji_zip, "r") as zip_ref:
        zip_ref.extractall(ruta_emojis)
    st.success("Emojis personalizados cargados correctamente.")
else:
    ruta_emojis = "emojis"  # carpeta emojis en tu repo

# Cargar emojis desde la ruta seleccionada
categorias, emoji_por_categoria = cargar_emojis(ruta_emojis)

# Subir imagen
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
        imagen_mostrar = imagen_bgr.copy()
        alto_imagen, ancho_imagen = imagen_mostrar.shape[:2]

        for i, box in enumerate(caras):
            if box.xyxy is None or len(box.xyxy) == 0:
                continue
            coords = box.xyxy[0]
            if len(coords) < 4:
                continue

            x1, y1, x2, y2 = map(int, coords)
            x1 = max(0, min(x1, ancho_imagen - 1))
            x2 = max(0, min(x2, ancho_imagen - 1))
            y1 = max(0, min(y1, alto_imagen - 1))
            y2 = max(0, min(y2, alto_imagen - 1))

            cv2.rectangle(imagen_mostrar, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pos_x = x1
            pos_y = y2 + 30
            if pos_y > alto_imagen - 10:
                pos_y = y2 - 10
            cv2.putText(imagen_mostrar, f"#{i}", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        st.image(cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB), caption="Caras detectadas (ID debajo)")

        # Selector múltiple de caras
        ids_seleccionados = st.multiselect(
            "Selecciona las caras a las que quieres aplicar emojis (IDs):",
            options=list(range(len(caras))),
            default=list(range(len(caras)))
        )

        # Selector de categoría de emojis
        categoria = st.selectbox("Selecciona la categoría de emojis:", categorias)

        # Botón para aplicar emojis
        if st.button("Aplicar emojis y mostrar preview"):
            imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(imagen_rgb)
            ruta_categoria = os.path.join(ruta_emojis, categoria)
            archivos_emoji = emoji_por_categoria[categoria]

            emojis_disponibles = archivos_emoji.copy()
            random.shuffle(emojis_disponibles)
            indice_emoji = 0

            for i, box in enumerate(caras):
                if i not in ids_seleccionados:
                    continue

                coords = box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords)
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

            # Botón para descargar
            buffered = io.BytesIO()
            imagen_pil.save(buffered, format="PNG")
            st.download_button(
                label="Descargar imagen con emojis",
                data=buffered.getvalue(),
                file_name="resultado_emojis.png",
                mime="image/png"
            )
