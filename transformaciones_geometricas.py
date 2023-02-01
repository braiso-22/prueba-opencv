import cv2 as cv
import matplotlib.pyplot as plt

from utils import Img
import numpy as np


def invertir(img):
    Img.mostrar_rgb(img)
    img2 = cv.flip(img, 1)
    Img.mostrar_rgb(img2)
    img2 = cv.flip(img2, 0)
    Img.mostrar_rgb(img2)
    # Invertir ambos ejes
    img2 = cv.flip(img2, -1)
    Img.mostrar_rgb(img2)


def invertir_matriz(img):
    Img.mostrar_rgb(img)
    img2 = cv.transpose(img)
    Img.mostrar_rgb(img2)


def resize(img: np.ndarray):
    Img.mostrar_rgb(img)
    img2: np.ndarray = cv.resize(img, (img.shape[0] * 2, img.shape[1]))
    Img.mostrar_rgb(img2)
    array_3_puntos = np.array([[0, 0], [0, img.shape[0] - 1], [img.shape[1] - 1, 0]], np.float32)
    array_3_puntos_2 = np.array([[img.shape[0] / 8, img.shape[1] / 8], [0, img.shape[0] / 2], [img.shape[1] / 2, 0]],
                                np.float32)
    plt.scatter(array_3_puntos[:, 0], array_3_puntos[:, 1], c='lime', s=120)
    plt.scatter(array_3_puntos_2[:, 0], array_3_puntos_2[:, 1], c='b', s=120)
    Img.mostrar_rgb(img)
    tranformacion = cv.getAffineTransform(array_3_puntos, array_3_puntos_2)
    img3 = cv.warpAffine(img, tranformacion, (img.shape[1], img.shape[0]))
    plt.scatter(array_3_puntos_2[:, 0], array_3_puntos_2[:, 1], c='b', s=120)
    Img.mostrar_rgb(img3)


def rotar(img: np.ndarray):
    Img.mostrar_rgb(img)
    img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    Img.mostrar_rgb(img2)
    centro = (img.shape[0] / 2, img.shape[1] / 2)
    angulo = 45
    size = 1
    rotacion = cv.getRotationMatrix2D(centro, angulo, size)
    img2 = cv.warpAffine(img, rotacion, (img.shape[0], img.shape[1]))
    Img.mostrar_rgb(img2)


def trasladar(img: np.ndarray):
    Img.mostrar_rgb(img)
    # 100 pixeles a la derecha y 50 pixeles hacia abajo
    matriz_traslacion = np.float32(
        [[1, 0, 100],
         [0, 1, 50]]
    )
    img2 = cv.warpAffine(img, matriz_traslacion, (img.shape[1], img.shape[0]))
    Img.mostrar_rgb(img2)


def main():
    chica = Img.cargar_imagen("img/chica.png")
    invertir(chica)
    invertir_matriz(chica)
    resize(chica)
    rotar(chica)
    trasladar(chica)
    pass


if __name__ == '__main__':
    main()
