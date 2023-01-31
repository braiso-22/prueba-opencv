import cv2
import matplotlib.pyplot as plt
import cv2 as cv

from utils import Img


def operaciones_bits_simples():
    rectangulo = Img.cargar_imagen("img/test5.png")
    circulo = Img.cargar_imagen("img/test6.png")

    circulo_gris = Img.escala_grises(circulo)
    rectangulo_gris = Img.escala_grises(rectangulo)

    operacion_and = cv.bitwise_and(circulo_gris, rectangulo_gris)
    operacion_or = cv.bitwise_or(circulo_gris, rectangulo_gris)
    operacion_xor = cv.bitwise_xor(circulo_gris, rectangulo_gris)
    operacion_not = cv.bitwise_not(circulo_gris)

    figura, (eje1, eje2, eje3, eje4, eje5, eje6) = plt.subplots(1, 6)
    eje1.imshow(circulo_gris, cmap="gray")
    eje2.imshow(rectangulo_gris, cmap="gray")
    eje3.imshow(operacion_and, cmap="gray")
    eje4.imshow(operacion_or, cmap="gray")
    eje5.imshow(operacion_xor, cmap="gray")
    eje6.imshow(operacion_not, cmap="gray")
    plt.show()


def main():
    operaciones_bits_simples()


if __name__ == '__main__':
    main()
