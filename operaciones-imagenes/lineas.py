from utils import Img
import cv2 as cv
import numpy as np


def detectar_lineas(image):
    gris = Img.escala_grises(image)
    Img.mostrar(gris, gris=True)
    blur = cv.blur(gris, (7, 7))
    Img.mostrar(blur, gris=True)
    canny = cv.Canny(blur, 0, 160)
    concidencias = int(75 * 4)
    lins = cv.HoughLinesP(
        image=canny,
        rho=7,
        theta=np.pi / 180,
        threshold=concidencias,
        minLineLength=45,
        maxLineGap=35
    )
    print(lins)
    Img.mostrar(canny, gris=True)
    image2 = image.copy()
    for linea in lins:
        x1, y1, x2, y2 = linea[0]
        image2 = cv.line(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)
    Img.mostrar(image2)


def main():
    image = Img.cargar_imagen("../img/carretera.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    Img.mostrar(image)
    detectar_lineas(image)

    pass


if __name__ == '__main__':
    main()
