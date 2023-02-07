import copy
import operaciones_bits
from utils import Img
import cv2 as cv
import numpy as np


def eliminar_mascara(image: np.ndarray, mask, nivel=10):
    # operaciones_bits.operaciones_bits_2_images(image, mask)
    copia = image.copy()
    mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)[1]
    not_mask = cv.bitwise_not(mask)

    image_con_agujeros = cv.bitwise_and(copia, not_mask)
    for i in range(nivel):
        # image_blur = cv.GaussianBlur(copia, (9, 9), sigmaX=0, sigmaY=0)

        filtro1 = np.ones((7, 7))
        filtro1[2:5, 2:5] = np.zeros((3, 3))
        filtro1 = filtro1 / 41
        
        image_blur = cv.filter2D(copia, ddepth=-1, kernel=filtro1)
        letras_difuminadas = Img.restar_imagenes(image_blur, image_con_agujeros)
        # letras_difuminadas = cv.bitwise_and(image_blur, mask)

        copia = Img.sumar_imagenes(image_con_agujeros, letras_difuminadas)

    final_mask = cv.bitwise_and(copia, mask)

    copia_mask1 = copy.deepcopy(final_mask)
    final_mask1 = cv.bitwise_xor(final_mask, Img.binarizar(copia_mask1, 200))

    result = cv.bitwise_xor(image, final_mask1)
    return result


def main():
    fondo = Img.cargar_imagen("img/fondo_paisaje.jpeg")
    texto = Img.cargar_imagen("img/texto_paisaje.jpeg")
    Img.mostrar_varios_rgb(fondo, texto)
    origen = Img.sumar_imagenes(fondo, texto)

    resultado = eliminar_mascara(origen, texto)
    Img.mostrar_varios_rgb(origen, resultado)


if __name__ == '__main__':
    main()
