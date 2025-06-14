import cv2
import os
import numpy as np
import random
from PIL import Image, ImageTk
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Configuración modelo
model_path = hf_hub_download(repo_id="sankn-videodb/yolo-v8-face", filename="yolov8m-face-lindevs.pt")
modelo = YOLO(model_path)

imagen_bgr = None
caras = []
categorias = []
emoji_por_categoria = {}
checkbox_vars = []

imagen_resultado_pil = None

# Carpeta emojis default (dentro del proyecto)
CARPETA_EMOJIS_DEFAULT = "emojis_default"

def cargar_emojis_default():
    global categorias, emoji_por_categoria
    if not os.path.exists(CARPETA_EMOJIS_DEFAULT):
        messagebox.showwarning("Aviso", f"No se encontró la carpeta por defecto de emojis: {CARPETA_EMOJIS_DEFAULT}")
        categorias = []
        emoji_por_categoria = {}
        return

    categorias = [d for d in os.listdir(CARPETA_EMOJIS_DEFAULT) if os.path.isdir(os.path.join(CARPETA_EMOJIS_DEFAULT, d))]
    emoji_por_categoria = {
        cat: [f for f in os.listdir(os.path.join(CARPETA_EMOJIS_DEFAULT, cat)) if f.lower().endswith('.png')]
        for cat in categorias
    }
    categoria_menu["values"] = categorias
    if categorias:
        categoria_menu.current(0)

def agregar_emojis_de_carpeta(ruta_carpeta):
    global categorias, emoji_por_categoria
    nuevas_categorias = [d for d in os.listdir(ruta_carpeta) if os.path.isdir(os.path.join(ruta_carpeta, d))]
    for cat in nuevas_categorias:
        ruta_cat = os.path.join(ruta_carpeta, cat)
        archivos = [f for f in os.listdir(ruta_cat) if f.lower().endswith('.png')]
        if cat in categorias:
            # Añadir sin duplicados
            existentes = set(emoji_por_categoria[cat])
            nuevos = [a for a in archivos if a not in existentes]
            emoji_por_categoria[cat].extend(nuevos)
        else:
            categorias.append(cat)
            emoji_por_categoria[cat] = archivos
    categoria_menu["values"] = categorias
    if categorias and categoria_menu.get() not in categorias:
        categoria_menu.current(0)

def seleccionar_carpeta_emojis():
    ruta = filedialog.askdirectory()
    if ruta:
        agregar_emojis_de_carpeta(ruta)

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
        label_preview.config(image="")  # Limpiar preview
        return

    # Mostrar imagen con rectángulos (sin IDs en la imagen)
    imagen_ids = imagen_bgr.copy()
    for i, box in enumerate(caras):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(imagen_ids, (x1, y1), (x2, y2), (0, 255, 0), 2)

    mostrar_preview(imagen_ids)

    # Crear checkboxes con texto debajo para seleccionar caras
    for i in range(len(caras)):
        var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(scrollable_frame, text=f"ID {i}", variable=var)
        cb.var = var
        cb.pack(anchor="w")
        checkbox_vars.append(cb)

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

    ruta_categoria_default = os.path.join(CARPETA_EMOJIS_DEFAULT, categoria)
    ruta_categoria_extra = None  # Si quieres manejar carpeta extra cargada, lo puedes agregar aquí

    archivos_emoji = emoji_por_categoria[categoria]
    if len(archivos_emoji) == 0:
        messagebox.showerror("Error", "No hay emojis en la categoría seleccionada.")
        return

    emojis_disponibles = archivos_emoji.copy()
    random.shuffle(emojis_disponibles)
    indice_emoji = 0

    for i, box in enumerate(caras):
        if not checkbox_vars[i].var.get():
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        ancho = x2 - x1
        alto = y2 - y1

        if indice_emoji >= len(emojis_disponibles):
            emojis_disponibles = archivos_emoji.copy()
            random.shuffle(emojis_disponibles)
            indice_emoji = 0

        emoji_path = os.path.join(CARPETA_EMOJIS_DEFAULT, categoria, emojis_disponibles[indice_emoji])
        indice_emoji += 1

        if not os.path.exists(emoji_path):
            messagebox.showerror("Error", f"No se encontró el archivo de emoji: {emoji_path}")
            return

        emoji_img = Image.open(emoji_path).convert("RGBA")
        escala_emoji = int(alto * 1.5)
        emoji_img = emoji_img.resize((escala_emoji, escala_emoji), Image.Resampling.LANCZOS)

        pos_x = x1 + (ancho - escala_emoji) // 2
        pos_y = y1 - escala_emoji // 5
        pos_y = max(pos_y, 0)

        imagen_pil.paste(emoji_img, (pos_x, pos_y), emoji_img)

    imagen_resultado_pil = imagen_pil

    # Mostrar imagen final con emojis (sin IDs)
    resultado = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Resultado final con emojis", resultado)

def guardar_imagen():
    global imagen_resultado_pil
    if imagen_resultado_pil is None:
        messagebox.showerror("Error", "Primero aplica los emojis para poder guardar.")
        return
    resultado = cv2.cvtColor(np.array(imagen_resultado_pil), cv2.COLOR_RGB2BGR)
    ruta_guardar = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG", "*.png")])
    if ruta_guardar:
        cv2.imwrite(ruta_guardar, resultado)
        messagebox.showinfo("Guardado", f"Imagen guardada en:\n{ruta_guardar}")

# Configuración ventana Tkinter
ventana = tk.Tk()
ventana.title("App de Emojis en Caras")

btn_cargar_img = tk.Button(ventana, text="Cargar Imagen", command=seleccionar_imagen)
btn_cargar_img.pack()

categoria_menu = ttk.Combobox(ventana, state="readonly")
categoria_menu.pack()

btn_cargar_emojis = tk.Button(ventana, text="Agregar carpeta emojis", command=seleccionar_carpeta_emojis)
btn_cargar_emojis.pack()

btn_aplicar = tk.Button(ventana, text="Aplicar emojis", command=aplicar_emojis_preview)
btn_aplicar.pack()

btn_guardar = tk.Button(ventana, text="Guardar imagen", command=guardar_imagen)
btn_guardar.pack()

label_preview = tk.Label(ventana)
label_preview.pack()

scrollable_frame = tk.Frame(ventana)
scrollable_frame.pack(fill="both", expand=True)

cargar_emojis_default()

ventana.mainloop()
