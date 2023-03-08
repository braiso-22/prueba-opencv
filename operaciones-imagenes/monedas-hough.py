from utils import Img
import cv2 as cv
import numpy as np


def main():
    image = Img.cargar_imagen("../img/monedas.png")

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image[90:850, 300:1200]
    image_copy = image.copy()

    image_gris = Img.escala_grises(image_copy)
    image_gris = cv.GaussianBlur(image_gris, (3, 3), sigmaX=0, sigmaY=0)
    circulos = cv.HoughCircles(
        image_gris,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=image_gris.shape[1] / 10,
        minRadius=35,
        maxRadius=500,
        param1=250,
        param2=29,
    )
    # print(circulos)
    if circulos is not None:
        circulos = np.uint32(circulos)
        areas = [int(np.pi * (i[2] ** 2)) for i in circulos[0, :]]
        area_maxima = np.max(areas)
        for i in circulos[0, :]:
            area = np.uint32(np.pi * (i[2] ** 2))
            cv.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv.putText(image, f"{area}", (i[0] + 5, i[1]),
                       cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(image, f"{set_valor(area_maxima, area)}", (i[0] + 5, i[1] + 30),
                       cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
    Img.mostrar(image)


def set_valor(area_maxima: float, area: float) -> str:
    euro2 = area_maxima
    euro1 = area_maxima * 0.9
    cent50 = area_maxima * 0.94
    cent20 = area_maxima * 0.86
    cent10 = area_maxima * 0.77
    cent5 = area_maxima * 0.82
    cent2 = area_maxima * 0.73
    cent1 = area_maxima * 0.63
    if area >= euro2:
        return '2 euros'
    elif area >= cent50:
        return '50c'
    elif area >= euro1:
        return '1€'
    elif area >= cent20:
        return '20c'
    elif area >= cent5:
        return '5c'
    elif area >= cent10:
        return '10c'
    elif area >= cent2:
        return '2c'
    elif area >= cent1:
        return '1c'
    else:
        return 'nose'


if __name__ == '__main__':
    main()
