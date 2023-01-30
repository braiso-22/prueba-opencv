import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


# En google colab # from google.colab.patches import cv2_imshow


def cargar_imagen(route: str = 'img/kong.jpg', window_name: str = 'prueba1'):
    src = cv.imread(route)
    cv.namedWindow("prueba1", cv.WINDOW_AUTOSIZE)
    cv.imshow("prueba1", src)
    cv.waitKey(0)
    return src


def print_image_info(image: np.ndarray):
    print(image.shape)
    print(image.size)
    print(image.dtype)
    print(image.ndim)


def separar_colores(image):
    red, green, blue = cv.split(image)
    # separar colores
    figura, (eje1, eje2, eje3) = plt.subplots(1, 3)
    eje1.imshow(red, cmap="gray", vmin=0, vmax=255)
    eje2.imshow(green, cmap="gray")
    eje3.imshow(blue, cmap="gray")

    plt.show()
    return red, green, blue


def unir_colores(red, green, blue):
    return cv.merge([blue, green, red])


def main():
    image = cargar_imagen()
    print_image_info(image)
    red, green, blue = separar_colores(image)
    image = unir_colores(red, green, blue)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
