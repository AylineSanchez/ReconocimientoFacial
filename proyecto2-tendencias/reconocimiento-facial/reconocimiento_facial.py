import cv2
import os
import face_recognition

# Ruta de las imágenes de rostros
imageFacesPath = "C:/Users/aylinealejandrasanch/Documents/GitHub/ReconocimientoFacial/rostros"

# Listas para almacenar las codificaciones de los rostros y sus nombres
facesEncodings = []
facesNames = []

# Iteramos sobre cada archivo de imagen en la carpeta de rostros
for file_name in os.listdir(imageFacesPath):
    # Cargamos la imagen y la convertimos de BGR a RGB
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculamos la codificación facial para la imagen
    f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
    
    # Agregamos la codificación y el nombre del archivo (sin extensión) a las listas correspondientes
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split(".")[0])

# Inicializamos la captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector facial utilizando el clasificador Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capturamos un fotograma de la cámara
    ret, frame = cap.read()
    
    # Salimos del bucle si no se pudo capturar un fotograma
    if ret == False:
        break
    
    # Volteamos horizontalmente el fotograma (para evitar que la imagen salga volteada)
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    
    # Detectamos rostros en el fotograma utilizando el clasificador Haar
    faces = faceClassif.detectMultiScale(frame, 1.1, 5)
    
    # Iteramos sobre cada rostro detectado en el fotograma
    for (x, y, w, h) in faces:
        # Recortamos el rostro de la imagen original y lo convertimos de BGR a RGB
        face = orig[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Calculamos la codificación facial para el rostro recortado
        actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
        
        # Comparamos la codificación del rostro actual con las codificaciones de rostros almacenadas previamente
        result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)
        
        # Determinamos el nombre del rostro actual
        if True in result:
            index = result.index(True)
            name = facesNames[index]
            color = (125, 220, 0)
        else:
            name = "Desconocido"
            color = (50, 50, 255)
        
        # Dibujamos un rectángulo alrededor del rostro y mostramos el nombre del rostro en el fotograma
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostramos el fotograma con los rectángulos y nombres de los rostros
    cv2.imshow("Frame", frame)
    
    # Esperamos una tecla y salimos del bucle si se presiona la tecla 'Esc'
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Liberamos la cámara y cerramos todas las ventanas al finalizar
cap.release()
cv2.destroyAllWindows()
