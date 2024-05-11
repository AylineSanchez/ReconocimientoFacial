import cv2
import os

# Ruta de las imágenes de entrada
imagesPath = "C:/Users/aylinealejandrasanch/Desktop/proyecto2-tendencias/reconocimiento-facial/imagenes/input"

# Comprobamos si la carpeta "rostros" existe, si no, la creamos
if not os.path.exists("rostros"):
    os.makedirs("rostros")
    print("Nueva carpeta: rostros")

# Inicializamos el clasificador de Haar para la detección facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Contador para nombrar los rostros extraídos
count = 0

# Iteramos sobre cada imagen en la carpeta de entrada
for imageName in os.listdir(imagesPath):
    print(imageName)
    # Cargamos la imagen
    image = cv2.imread(imagesPath + "/" + imageName)
    # Detectamos los rostros en la imagen
    faces = faceClassif.detectMultiScale(image, 1.1, 5)
    
    # Iteramos sobre cada rostro detectado
    for (x, y, w, h) in faces:
        # Dibujamos un rectángulo alrededor del rostro en la imagen original
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Recortamos el rostro de la imagen original
        face = image[y:y + h, x:x + w]
        # Redimensionamos el rostro a 150x150 píxeles
        face = cv2.resize(face, (150, 150))
        # Guardamos el rostro recortado en la carpeta "rostros"
        cv2.imwrite("rostros/" + str(count) + ".jpg", face)
        count += 1
        # Mostramos el rostro recortado en una ventana
        cv2.imshow("face", face)
        cv2.waitKey(0)
    
    # Mostramos la imagen original con los rectángulos dibujados alrededor de los rostros detectados
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Cerramos todas las ventanas al finalizar
cv2.destroyAllWindows()
