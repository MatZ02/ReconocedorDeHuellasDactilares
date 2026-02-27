import cv2
import os
import random
import time
import numpy as np



RUTA_DATASET = r"C:\Users\JUAN\OneDrive\Escritorio\pratica web\Huellas"   
RATIO_TEST = 0.75         
EXTENSIONES_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def cargar_imagenes(ruta):
    print("Buscando en:", ruta)

    if not os.path.exists(ruta):
        print("La carpeta NO existe")
        return [], []

    archivos = os.listdir(ruta)
    print(f"Archivos encontrados: {len(archivos)}")

    imagenes = []
    nombres = []

    for archivo in archivos:
        print("   -", archivo)

        if archivo.lower().endswith(EXTENSIONES_VALIDAS):
            img = cv2.imread(os.path.join(ruta, archivo), 0)

            if img is not None:
                imagenes.append(img)
                nombres.append(archivo)

    print(f"Imágenes válidas cargadas: {len(imagenes)}")
    return imagenes, nombres

def preprocesamiento(img):
    img = cv2.resize(img, (300, 300))
    img = cv2.GaussianBlur(img, (5,5), 0)

    binaria = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return binaria


def extraer_orb(img):
    orb = cv2.ORB_create(1000)
    keypoints, descriptores = orb.detectAndCompute(img, None)
    return keypoints, descriptores


def comparar(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)

    buenos = []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            buenos.append(m)

    return len(buenos)


def main():

    print("\n Analizando base de datos de huellas dactilares...")
    imagenes, nombres = cargar_imagenes(RUTA_DATASET)

    if len(imagenes) == 0:
        print("No se encontraron imágenes.")
        return

    print(f"{len(imagenes)} imágenes cargadas")

    indice_random = random.randint(0, len(imagenes)-1)
    img_consulta = imagenes[indice_random]
    nombre_real = nombres[indice_random]

    print(f"\nHuella desconocida seleccionada: {nombre_real}")

    img_consulta = preprocesamiento(img_consulta)
    kp1, des1 = extraer_orb(img_consulta)

    mejor_match = 0
    mejor_usuario = ""

    inicio = time.time()

    for img, nombre in zip(imagenes, nombres):

        img = preprocesamiento(img)
        kp2, des2 = extraer_orb(img)

        score = comparar(des1, des2)

        if score > mejor_match:
            mejor_match = score
            mejor_usuario = nombre

    fin = time.time()

    tiempo = fin - inicio

    print("\n================ RESULTADOS =================")
    print(f"Usuario identificado: {mejor_usuario}")
    print(f"Puntaje de coincidencia: {mejor_match}")
    print(f"tiempo de ejecución: {tiempo:.4f} segundos")
    print("============================================")


if __name__ == "__main__":
    main()

#Juan Jose Londono Martinez :D
