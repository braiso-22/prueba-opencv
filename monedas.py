from utils import Img
import cv2 as cv
import numpy as np


def main():
    image = Img.cargar_imagen("img/monedas.png")
    r, g, b = Img.separar_rgb(image)
    h, s, v = Img.separar_hsv(image)
    l, u, v = Img.separar_luv(image)
    # Img.mostrar_varios_rgb(b, s, u)

    binarizado_b = Img.binarizar(b, 128)
    binarizado_s = Img.binarizar(s, 45)
    suma = cv.bitwise_and(binarizado_b, binarizado_s)
    Img.mostrar_varios_rgb(binarizado_b, binarizado_s, suma)
    kernel = np.ones((3, 3), np.uint8)

    abrir = cv.morphologyEx(suma, cv.MORPH_OPEN, kernel, iterations=3)
    cerrar = cv.morphologyEx(abrir, cv.MORPH_CLOSE, kernel, iterations=3)
    Img.mostrar_varios_rgb(suma, abrir, cerrar)


if __name__ == '__main__':
    main()
