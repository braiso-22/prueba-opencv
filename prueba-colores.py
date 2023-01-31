from utils import Img
import cv2 as cv
import numpy as np


def main():
    image = Img.cargar_imagen()
    Img.print_image_info(image)
    Img.mostrar(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    rojo_bajo_1 = np.array([0, 155, 120], np.uint8)
    rojo_alto_1 = np.array([10, 255, 255], np.uint8)
    rojo_bajo_2 = np.array([165, 100, 170], np.uint8)
    rojo_alto_2 = np.array([179, 255, 255], np.uint8)
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_rojo_1 = cv.inRange(image_hsv, rojo_bajo_1, rojo_alto_1)
    mask_rojo_2 = cv.inRange(image_hsv, rojo_bajo_2, rojo_alto_2)
    mask_rojo = cv.add(mask_rojo_1, mask_rojo_2)

    mask = cv.bitwise_and(image_hsv, image_hsv, mask=mask_rojo)
    Img.mostrar(cv.cvtColor(mask, cv.COLOR_HSV2RGB))

    Img.mostrar(cv.cvtColor(mask_rojo, cv.COLOR_GRAY2RGB))


if __name__ == '__main__':
    main()
