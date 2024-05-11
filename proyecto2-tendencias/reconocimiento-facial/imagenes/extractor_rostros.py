import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Ruta de las imágenes de entrada
imagesPath = "C:/Users/aylinealejandrasanch/Documents/GitHub/ReconocimientoFacial/proyecto2-tendencias/reconocimiento-facial/imagenes/input"

# Función para cargar una imagen a la carpeta de entrada
def cargar_imagen():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Copiar la imagen a la carpeta de entrada
        os.replace(file_path, os.path.join(imagesPath, os.path.basename(file_path)))
        print("Imagen cargada con éxito.")

# Función para mostrar todos los rostros encontrados y guardarlos en la carpeta "rostros"
def mostrar_rostros():
    # Inicializamos una lista para almacenar todos los rostros
    all_faces = []
    # Iteramos sobre cada imagen en la carpeta de entrada
    for imageName in os.listdir(imagesPath):
        # Cargamos la imagen
        image = cv2.imread(os.path.join(imagesPath, imageName))
        # Detectamos los rostros en la imagen
        faces = faceClassif.detectMultiScale(image, 1.1, 5)
        # Iteramos sobre cada rostro detectado
        for (x, y, w, h) in faces:
            # Recortamos el rostro de la imagen original
            face = image[y:y + h, x:x + w]
            # Redimensionamos el rostro a 150x150 píxeles
            face = cv2.resize(face, (150, 150))
            # Guardamos el rostro recortado en la carpeta "rostros"
            cv2.imwrite("rostros/" + str(len(all_faces)) + ".jpg", face)
            # Agregamos el rostro a la lista de rostros
            all_faces.append(face)
    # Creamos una ventana para mostrar todos los rostros
    all_faces_window = tk.Toplevel(root)
    for i, face in enumerate(all_faces):
        # Convertimos la imagen de OpenCV a formato compatible con Tkinter
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Convertimos la imagen a un objeto PhotoImage de Tkinter
        face_tk = tk.PhotoImage(data=cv2.imencode('.png', face_rgb)[1].tobytes())
        # Mostramos la imagen en la ventana
        label = tk.Label(all_faces_window, image=face_tk)
        label.image = face_tk
        label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
        
# Inicializamos el clasificador de Haar para la detección facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Creamos la ventana principal
root = tk.Tk()
root.title("Detector de rostros")

# Botón para cargar una imagen a la carpeta de entrada
cargar_button = tk.Button(root, text="Cargar Imagen", command=cargar_imagen)
cargar_button.pack(pady=10)

# Botón para mostrar todos los rostros encontrados
mostrar_button = tk.Button(root, text="Mostrar Rostros", command=mostrar_rostros)
mostrar_button.pack(pady=10)

# Mantenemos la ventana abierta
root.mainloop()
