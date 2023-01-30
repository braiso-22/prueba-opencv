import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from utils import Img


def main():
    imagen = Img.cargar_imagen("img/banda_caja1.png")
    r, g, b = Img.separar_rgb(imagen)
    h, s, v = Img.separar_hsv(imagen)
    l, u, v = Img.separar_luv(imagen)


if __name__ == '__main__':
    main()
