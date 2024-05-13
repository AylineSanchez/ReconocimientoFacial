import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Función para cargar una imagen y mostrar los rostros encontrados
def mostrar_rostros(image_path):
    # Verificar si la ruta es válida y si la imagen existe
    if os.path.exists(image_path) and os.path.isfile(image_path):
        # Cargar la imagen
        image = cv2.imread(image_path)
        # Detectar rostros en la imagen
        faces = faceClassif.detectMultiScale(image, 1.1, 5)
        # Si se detectan rostros, mostrar cada rostro en una ventana emergente
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                # Recortar el rostro de la imagen original
                face = image[y:y + h, x:x + w]
                # Redimensionar el rostro a 150x150 píxeles
                face = cv2.resize(face, (150, 150))
                # Convertir la imagen de OpenCV a formato compatible con Tkinter
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                # Convertir la imagen a un objeto PhotoImage de Tkinter
                face_tk = tk.PhotoImage(data=cv2.imencode('.png', face_rgb)[1].tobytes())
                # Crear una ventana para mostrar el rostro y solicitar el nombre
                select_name_window = tk.Toplevel(root)
                select_name_window.title("Seleccionar nombre del rostro {}".format(i+1))
                # Mostrar la imagen en la ventana
                label = tk.Label(select_name_window, image=face_tk)
                label.image = face_tk
                label.pack(padx=5, pady=5)
                # Variable de control para el nombre del rostro
                name_var = tk.StringVar()
                # Solicitar al usuario que ingrese un nombre para el rostro
                name_label = tk.Label(select_name_window, text="Ingrese el nombre del rostro:")
                name_label.pack(padx=5, pady=5)
                name_entry = tk.Entry(select_name_window, textvariable=name_var)
                name_entry.pack(padx=5, pady=5)
                # Función para guardar el rostro con el nombre ingresado
                def save_face():
                    name = name_var.get()
                    if name:
                        # Guardar el rostro recortado en la carpeta "rostros"
                        cv2.imwrite(os.path.join("rostros", name + ".jpg"), face)
                        # Cerrar la ventana después de guardar
                        select_name_window.destroy()
                    else:
                        messagebox.showerror("Error", "Por favor, ingrese un nombre para el rostro.")
                save_button = tk.Button(select_name_window, text="Guardar", command=save_face)
                save_button.pack(padx=5, pady=5)
                # Esperar hasta que el usuario cierre la ventana antes de continuar con el siguiente rostro
                select_name_window.wait_window()
        else:
            print("No se detectaron rostros en la imagen.")
    else:
        print("La ruta de la imagen no es válida o la imagen no existe.")

# Función para cargar una imagen desde el sistema de archivos
def cargar_imagen():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Llamar a la función mostrar_rostros() con la imagen seleccionada
        mostrar_rostros(file_path)

# Inicializar el clasificador de Haar para la detección facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Crear la ventana principal de la aplicación
root = tk.Tk()
root.title("Detector de rostros")

# Botón para cargar una imagen desde el sistema de archivos
cargar_button = tk.Button(root, text="Cargar Imagen", command=cargar_imagen)
cargar_button.pack(pady=10)

# Mantener la ventana abierta
root.mainloop()
