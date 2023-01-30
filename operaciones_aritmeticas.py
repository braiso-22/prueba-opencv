import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from utils import Img


def main():
    imagen = Img.cargar_imagen("img/banda_vacia.png")
    imagen1 = Img.cargar_imagen("img/banda_caja1.png")
    imagen2 = Img.cargar_imagen("img/banda_caja1_5.png")
    imagen3 = Img.cargar_imagen("img/banda_caja2.png")
    r, g, b = Img.separar_rgb(imagen1)
    h, s, v = Img.separar_hsv(imagen1)
    l, u, v = Img.separar_luv(imagen1)

    img = Img.escala_grises(imagen)
    img1 = Img.escala_grises(imagen1)
    img2 = Img.escala_grises(imagen2)
    img3 = Img.escala_grises(imagen3)


if __name__ == '__main__':
    main()
