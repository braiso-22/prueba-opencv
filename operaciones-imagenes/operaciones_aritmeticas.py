import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from utils import Img


def mostrar_imagenes():
    imagen = Img.cargar_imagen("../img/banda_vacia.png")
    imagen1 = Img.cargar_imagen("../img/banda_caja1.png")
    imagen2 = Img.cargar_imagen("../img/banda_caja1_5.png")
    imagen3 = Img.cargar_imagen("../img/banda_caja2.png")
    # Img.ver_todos_los_grises(imagen1)
    img = Img.escala_grises(imagen)
    img1 = Img.escala_grises(imagen1)
    img2 = Img.escala_grises(imagen2)
    img3 = Img.escala_grises(imagen3)

    return img, img1, img2, img3


def main():
    array = mostrar_imagenes()

    Img.restar_imagenes(array[0], array[1])
    resta1 = Img.restar_imagenes(array[1], array[0])
    Img.restar_imagenes(array[1], array[2])
    resta2 = Img.restar_imagenes(array[2], array[0])
    Img.restar_imagenes(array[2], array[3])
    resta3 = Img.restar_imagenes(array[3], array[0])
    Img.sumar_imagenes(resta1, resta2)
    Img.sumar_imagenes(resta2, resta3)
    Img.multiplicar_imagenes(resta1, array[0])
    Img.dividir_imanenes(array[1], array[0])


if __name__ == '__main__':
    main()
