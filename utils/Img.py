import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def cargar_imagen(route: str = 'img/kong.jpg', window_name: str = 'prueba1', mostrar=False):
    src = cv.imread(route)
    if mostrar:
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(window_name, src)
        cv.waitKey(0)
    return src


def print_image_info(image: np.ndarray):
    print(image.shape)
    print(image.size)
    print(image.dtype)
    print(image.ndim)


def show_3_gray(one, two, three):
    figura, (eje1, eje2, eje3) = plt.subplots(1, 3)
    eje1.imshow(one, cmap="gray", vmin=0, vmax=255)
    eje2.imshow(two, cmap="gray")
    eje3.imshow(three, cmap="gray")
    plt.show()


def separar_rgb(image):
    red, green, blue = cv.split(image)
    # separar colores
    show_3_gray(red, green, blue)
    return red, green, blue


def separar_hsv(image):
    src = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h = src[:, :, 0]
    s = src[:, :, 1]
    v = src[:, :, 2]
    show_3_gray(h, s, v)
    return h, s, v


def separar_luv(image):
    src = cv.cvtColor(image, cv.COLOR_BGR2Luv)
    h = src[:, :, 0]
    s = src[:, :, 1]
    v = src[:, :, 2]
    show_3_gray(h, s, v)
    return h, s, v
    src


def unir_colores(red, green, blue):
    return cv.merge([blue, green, red])


def escala_grises(img, mostrar=False):
    grises = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if mostrar:
        mostrar(grises, gris=True)
    return grises


def ver_todos_los_grises(img):
    r, g, b = separar_rgb(img)
    h, s, v = separar_hsv(img)
    l, u, v = separar_luv(img)


def restar_imagenes(img, img2):
    """
    Resta con saturación
    :param img:
    :param img2:
    :return:
    """
    resta = cv.subtract(img, img2)
    mostrar(resta, gris=True)
    return resta


def sumar_imagenes(img, img2):
    suma = cv.add(img, img2)
    mostrar(suma, gris=True)
    return suma


def multiplicar_imagenes(img, img2):
    multiplicacion = cv.multiply(img, img2)
    mostrar(multiplicacion, gris=True)
    return multiplicacion


def dividir_imanenes(img, img2):
    division = cv.divide(img, img2)
    mostrar(division, gris=True)
    return division


def ver_cambios_entre(img1, img2):
    caja1 = escala_grises(
        img1
    )
    caja2 = escala_grises(
        img2
    )
    cambios = cv.bitwise_xor(caja1, caja2)
    show_3_gray(caja1, caja2, cambios)


def mostrar(img, gris=False):
    if gris:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.show()


def binarizar(img, media):
    img[img < media] = 0
    img[img >= media] = 255
    return img
