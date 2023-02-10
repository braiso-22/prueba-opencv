from utils import Img
import cv2 as cv
import numpy as np
import math


def main():
    image = Img.cargar_imagen("img/monedas.png")
    r, g, b = Img.separar_rgb(image)
    h, s, v = Img.separar_hsv(image)
    l, u, v = Img.separar_luv(image)
    # Img.mostrar_varios_rgb(b, s, u)
    kernel = np.ones((3, 3), np.uint8)
    b2 = cv.GaussianBlur(b, (5, 5), 0)

    binarizado_b = Img.binarizar(b2, 128)
    binarizado_b = cv.GaussianBlur(binarizado_b, (5, 5), 0)
    # Img.mostrar(binarizado_b, gris=True)
    binarizado_b = cv.morphologyEx(binarizado_b, cv.MORPH_CLOSE, kernel, iterations=4)
    binarizado_b = cv.morphologyEx(binarizado_b, cv.MORPH_OPEN, kernel, iterations=6)

    binarizado_u = Img.binarizar(u, 102)
    binarizado_u = cv.GaussianBlur(binarizado_u, (5, 5), 0)
    # Img.mostrar(binarizado_u, gris=True)
    binarizado_u = cv.morphologyEx(binarizado_u, cv.MORPH_CLOSE, kernel, iterations=2)
    binarizado_u = cv.morphologyEx(binarizado_u, cv.MORPH_OPEN, kernel, iterations=20)

    suma = cv.bitwise_and(binarizado_b, binarizado_u)
    suma = cv.GaussianBlur(suma, (3, 3), 0)
    suma = Img.binarizar(suma, 130)
    # Img.mostrar(suma, gris=True)
    suma = cv.morphologyEx(suma, cv.MORPH_CLOSE, kernel, iterations=5)
    suma = cv.morphologyEx(suma, cv.MORPH_OPEN, kernel, iterations=15)

    # Img.mostrar_varios_rgb(binarizado_b, binarizado_u, suma)

    suma = cv.bitwise_not(suma)
    # suma = cv.GaussianBlur(suma, (31, 31), 0)
    suma = cv.erode(suma, kernel, iterations=8)
    suma = cv.dilate(suma, kernel, iterations=10)
    suma = cv.morphologyEx(suma, cv.MORPH_CLOSE, kernel, iterations=20)
    # Img.mostrar(suma, gris=True)

    umbral_minimo = 50
    umbral_maximo = 100

    canny: np.ndarray = cv.Canny(suma, umbral_minimo, umbral_maximo)
    # Img.mostrar(canny, gris=True)

    contornos, jerarquia = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("Monedas encontradas: {}".format(len(contornos)))

    # dibujar todos los contornos
    # cv.drawContours(cv.cvtColor(monedas_rodeadas, cv.COLOR_BGR2RGB), contornos, -1, (0, 255, 0), 2)

    monedas_rodeadas = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)
    perimetros = [cv.arcLength(contorno, True) for contorno in contornos]
    centros = [cv.moments(contorno) for contorno in contornos]

    for i in range(len(contornos)):
        monedas_rodeadas = cv.circle(
            monedas_rodeadas, (
                int(centros[i]['m10'] / centros[i]['m00']),
                int(centros[i]['m01'] / centros[i]['m00'])
            ),
            int(perimetros[i] / (2 * math.pi)),
            (0, 255, 0), 2)

    Img.mostrar(monedas_rodeadas)


if __name__ == '__main__':
    main()
