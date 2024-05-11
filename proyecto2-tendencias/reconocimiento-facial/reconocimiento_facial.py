import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import face_recognition
import subprocess

# Ruta de las imágenes de rostros
imageFacesPath = "C:/Users/aylinealejandrasanch/Documents/GitHub/ReconocimientoFacial/rostros"

# Lista para almacenar las codificaciones de los rostros conocidos
known_faces_encodings = []

# Función para cargar las imágenes de rostros al inicio
def cargar_imagenes_iniciales():
    global known_faces_encodings

    # Limpiar la lista de codificaciones de rostros conocidos
    known_faces_encodings.clear()

    # Iterar sobre cada archivo de imagen en la carpeta de rostros
    for file_name in os.listdir(imageFacesPath):
        # Cargar la imagen y convertirla de BGR a RGB
        image = cv2.imread(os.path.join(imageFacesPath, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calcular la codificación facial para la imagen
        f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
        
        # Agregar la codificación a la lista de codificaciones de rostros conocidos
        known_faces_encodings.append(f_coding)

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector facial utilizando el clasificador Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Función para actualizar el frame en la interfaz
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        orig = frame.copy()

        # Detectar rostros en el fotograma utilizando el clasificador Haar
        faces = faceClassif.detectMultiScale(frame, 1.1, 5)

        # Iterar sobre cada rostro detectado en el fotograma
        for (x, y, w, h) in faces:
            face = orig[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Calcular la codificación facial para el rostro recortado
            actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]

            # Comparar la codificación del rostro actual con las codificaciones de rostros conocidos
            result = face_recognition.compare_faces(known_faces_encodings, actual_face_encoding)

            # Determinar el nombre del rostro actual
            if True in result:
                index = result.index(True)
                name = os.listdir(imageFacesPath)[index].split(".")[0]
                color = (125, 220, 0)
            else:
                name = "Desconocido"
                color = (50, 50, 255)

            # Dibujar un rectángulo alrededor del rostro y mostrar el nombre del rostro en el fotograma
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convertir el frame a formato compatible con Tkinter
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar la imagen en el label
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # Llamar a la función después de un breve intervalo
    label.after(10, update_frame)

# Función para abrir la interfaz anterior
def abrir_interfaz_anterior():
    subprocess.Popen(["python", "C:/Users/aylinealejandrasanch/Documents/GitHub/ReconocimientoFacial/proyecto2-tendencias/reconocimiento-facial/imagenes/extractor_rostros.py"], shell=True)

# Cargar las imágenes de rostros al inicio
cargar_imagenes_iniciales()

# Crear ventana
root = tk.Tk()
root.title("Interfaz de Reconocimiento Facial")

# Crear un label para mostrar la imagen de la cámara
label = tk.Label(root)
label.pack()

# Botón para abrir la interfaz anterior
abrir_button = tk.Button(root, text="Abrir Interfaz Anterior", command=abrir_interfaz_anterior)
abrir_button.pack(pady=10)

# Ocultar la ventana principal
root.iconify()

# Ejecutar la función para actualizar el frame
update_frame()

# Ejecutar la aplicación
root.mainloop()
