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


def main():
    image = cargar_imagen()
    print_image_info(image)


if __name__ == "__main__":
    main()
