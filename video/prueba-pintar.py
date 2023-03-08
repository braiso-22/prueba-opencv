import cv2 as cv
import numpy as np
from utils import Img


def pintar_en_fondo_negro():
    img = np.zeros((512, 512, 3), np.uint8)
    cv.putText(img, "Hola", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    cv.rectangle(img, (384, 6), (510, 128), (0, 255, 0), 3)
    cv.circle(img, (447, 63), 55, (0, 0, 255), -1)
    Img.mostrar(img)


def pintar_en_la_j():
    j = Img.cargar_imagen("../img/j0.png")

    kernel = np.ones((5, 5), np.uint8)

    j2 = cv.erode(j, kernel, iterations=1)
    j3 = cv.dilate(j, kernel, iterations=1)
    j4 = cv.morphologyEx(j, cv.MORPH_OPEN, kernel)
    j5 = cv.morphologyEx(j, cv.MORPH_CLOSE, kernel)
    j6 = cv.morphologyEx(j, cv.MORPH_GRADIENT, kernel)
    j7 = cv.morphologyEx(j2, cv.MORPH_OPEN, kernel)
    j8 = cv.erode(j4, kernel, iterations=1)
    Img.mostrar_varios_rgb(j, j2, j3, j4, j5, j6, j7, j8)


def limpiar_fondo_j():
    kernel = np.ones((5, 5), np.uint8)
    j_sucia = Img.cargar_imagen("../img/j1.png")
    j_fina = cv.erode(j_sucia, kernel, iterations=1)
    j_limpia = cv.dilate(j_fina, kernel, iterations=1)
    Img.mostrar_varios_rgb(j_sucia, j_limpia)


def limpiar_j_sucia():
    kernel = np.ones((5, 5), np.uint8)
    j_sucia = Img.cargar_imagen("../img/j2.png")
    j_gorda = cv.dilate(j_sucia, kernel, iterations=1)
    j_limpia = cv.erode(j_gorda, kernel, iterations=1)
    Img.mostrar_varios_rgb(j_sucia, j_limpia)


def limpiar_letras():
    kernel = np.ones((5, 5), np.uint8)
    letras = Img.cargar_imagen("../img/letras.png")
    letras_cerrar = cv.morphologyEx(letras, cv.MORPH_CLOSE, kernel, iterations=4)
    letras_abrir = cv.morphologyEx(letras_cerrar, cv.MORPH_OPEN, kernel, iterations=3)

    Img.mostrar_varios_rgb(letras, letras_cerrar, letras_abrir)


def main():
    pintar_en_fondo_negro()
    pintar_en_la_j()
    limpiar_fondo_j()
    limpiar_j_sucia()
    limpiar_letras()
    pass


if __name__ == '__main__':
    main()
