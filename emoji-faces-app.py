# %%
# %pip install pyinstaller

# %%
import cv2
import os
import numpy as np
import random
from PIL import Image, ImageTk
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------- CONFIGURACIÓN ----------------

model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
modelo = YOLO(model_path)

imagen_bgr = None
caras = []
carpeta_emojis = ""
categorias = []
emoji_por_categoria = {}
checkbox_vars = []

imagen_resultado_pil = None  # Variable global para guardar la imagen con emojis aplicada

# ---------------- FUNCIONES ----------------

def seleccionar_imagen():
    global imagen_bgr
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
    if not ruta:
        return
    imagen = cv2.imread(ruta)
    if imagen is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        return

    alto_max = 800
    if imagen.shape[0] > alto_max:
        escala = alto_max / imagen.shape[0]
        imagen = cv2.resize(imagen, (int(imagen.shape[1] * escala), alto_max))

    imagen_bgr = imagen
    detectar_caras()

def seleccionar_carpeta_emojis():
    global carpeta_emojis, categorias, emoji_por_categoria
    carpeta_emojis = filedialog.askdirectory()
    if carpeta_emojis:
        categorias = [d for d in os.listdir(carpeta_emojis) if os.path.isdir(os.path.join(carpeta_emojis, d))]
        emoji_por_categoria = {
            cat: [f for f in os.listdir(os.path.join(carpeta_emojis, cat)) if f.endswith('.png')]
            for cat in categorias
        }
        categoria_menu["values"] = categorias
        if categorias:
            categoria_menu.current(0)

def detectar_caras():
    global caras, checkbox_vars
    if imagen_bgr is None:
        return

    resultados = modelo(imagen_bgr)
    caras = resultados[0].boxes

    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    checkbox_vars.clear()

    if len(caras) == 0:
        messagebox.showinfo("Sin caras", "No se detectaron caras.")
        return

    # Mostrar la imagen con los IDs
    imagen_ids = imagen_bgr.copy()
    for i, box in enumerate(caras):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(imagen_ids, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen_ids, f"ID {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Crear checkbox
        var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(scrollable_frame, text=f"ID {i}", variable=var)
        cb.var = var
        cb.pack(anchor="w")
        checkbox_vars.append(cb)

    # Mostrar preview
    mostrar_preview(imagen_ids)

def mostrar_preview(imagen):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    imagen_pil = imagen_pil.resize((450, int(imagen.shape[0] * (450 / imagen.shape[1]))), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(imagen_pil)

    label_preview.config(image=imgtk)
    label_preview.image = imgtk

def aplicar_emojis_preview():
    global imagen_resultado_pil
    if imagen_bgr is None or not caras:
        messagebox.showerror("Error", "Primero selecciona una imagen y detecta caras.")
        return

    categoria = categoria_menu.get()
    if not categoria or categoria not in emoji_por_categoria:
        messagebox.showerror("Error", "Categoría no válida o sin emojis.")
        return

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)

    ruta_categoria = os.path.join(carpeta_emojis, categoria)
    archivos_emoji = emoji_por_categoria[categoria]

    # Preparar lista de emojis barajada
    emojis_disponibles = archivos_emoji.copy()
    random.shuffle(emojis_disponibles)
    indice_emoji = 0

    for i, box in enumerate(caras):
        if not checkbox_vars[i].var.get():
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ancho = x2 - x1
        alto = y2 - y1

        # Si se acabaron los emojis, reiniciar y barajar de nuevo
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

    imagen_resultado_pil = imagen_pil  # Guardar para posterior guardado

    resultado = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Vista previa con emojis", resultado)

def guardar_imagen():
    global imagen_resultado_pil
    if imagen_resultado_pil is None:
        messagebox.showerror("Error", "Primero aplica los emojis para poder guardar.")
        return

    resultado = cv2.cvtColor(np.array(imagen_resultado_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite("resultado_final_emojis_app.jpg", resultado)
    messagebox.showinfo("Guardado", "La imagen ha sido guardada como 'resultado_final_emojis_app.jpg'")

# ---------------- INTERFAZ ----------------

root = tk.Tk()
root.title("👾 App de Emojis en Caras")
root.geometry("600x900")

# --- Primera fila de botones y combobox ---

frame_fila1 = tk.Frame(root)
frame_fila1.pack(pady=10, fill="x")

btn_imagen = tk.Button(frame_fila1, text="📷 Seleccionar Imagen", command=seleccionar_imagen, width=20)
btn_imagen.grid(row=0, column=0, padx=5)

btn_emojis = tk.Button(frame_fila1, text="📁 Seleccionar Carpeta de Emojis", command=seleccionar_carpeta_emojis, width=25)
btn_emojis.grid(row=0, column=1, padx=5)

tk.Label(frame_fila1, text="🎨 Categoría de Emojis:").grid(row=0, column=2, padx=5)
categoria_menu = ttk.Combobox(frame_fila1, state="readonly", width=15)
categoria_menu.grid(row=0, column=3, padx=5)

# --- Segunda fila de botones ---

frame_fila2 = tk.Frame(root)
frame_fila2.pack(pady=5, fill="x")

btn_aplicar_preview = tk.Button(frame_fila2, text="👁️ Aplicar Emojis (Preview)", command=aplicar_emojis_preview, bg="lightblue", width=25)
btn_aplicar_preview.grid(row=0, column=0, padx=10)

btn_guardar = tk.Button(frame_fila2, text="💾 Guardar Imagen", command=guardar_imagen, bg="lightgreen", width=25)
btn_guardar.grid(row=0, column=1, padx=10)

# Preview de imagen
label_preview = tk.Label(root)
label_preview.pack(pady=10)

# Scrollable frame para los IDs
frame_scroll_ids = tk.Frame(root)
frame_scroll_ids.pack(pady=10, fill="both", expand=True)

canvas_ids = tk.Canvas(frame_scroll_ids, height=200)
scrollbar_ids = tk.Scrollbar(frame_scroll_ids, orient="vertical", command=canvas_ids.yview)
scrollable_frame = tk.Frame(canvas_ids)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_ids.configure(
        scrollregion=canvas_ids.bbox("all")
    )
)

canvas_ids.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas_ids.configure(yscrollcommand=scrollbar_ids.set)

canvas_ids.pack(side="left", fill="both", expand=True)
scrollbar_ids.pack(side="right", fill="y")

root.mainloop()



