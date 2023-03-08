import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from utils import Img


# En google colab # from google.colab.patches import cv2_imshow


def main():
    image = Img.cargar_imagen()
    Img.print_image_info(image)
    red, green, blue = Img.separar_rgb(image)
    image = Img.unir_colores(red, green, blue)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
