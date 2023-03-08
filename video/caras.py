import cv2 as cv
import numpy as np
from utils import Img
from utils import Camera


def __detectar(image: np.ndarray, clasifier: str) -> np.ndarray:
    copy = image.copy()
    gris = Img.escala_grises(copy)
    detector = cv.CascadeClassifier(cv.data.haarcascades + clasifier)
    buscado = detector.detectMultiScale(gris, 1.25, 5)

    return buscado


def detectar_bocas(image: np.ndarray):
    bocas = "haarcascade_smile.xml"
    return __detectar(image, bocas)


def detectar_ojos(image: np.ndarray):
    ojos = "haarcascade_eye.xml"
    return __detectar(image, ojos)


def detectar_caras(image: np.ndarray):
    caras = "haarcascade_frontalface_default.xml"
    return __detectar(image, caras)


def pintar_bordes(image: np.ndarray, bordes: np.ndarray, color: tuple = (10, 255, 10)) -> np.ndarray:
    copy = image.copy()
    for (x, y, w, h) in bordes:
        cv.rectangle(copy, (x, y), (x + w, y + h), color, 2)
    return copy


def detectar_cosas(image, params=None):
    caras = detectar_caras(image)
    ojos = detectar_ojos(image)
    sonrisas = detectar_bocas(image)
    sonrisas1 = np.empty((0, 4), int)
    ojos1 = np.empty((0, 4), int)
    # RGB
    verde = (0, 255, 0)
    rojo = (255, 0, 0)
    azul = (0, 0, 255)

    # obtener los ojos dentro de una cara
    for cara in caras:
        x, y, w, h = cara
        for (x2, y2, w2, h2) in ojos:
            if x < x2 < x + w and y < y2 < y + h:
                ojos1 = np.append(ojos1, [[x2, y2, w2, h2]], axis=0)

    for (x, y, w, h) in caras:
        for (x2, y2, w2, h2) in sonrisas:
            if x < x2 < (x + w) and y < y2 < (y + h):
                sonrisas1 = np.append(sonrisas1, [[x2, y2, w2, h2]], axis=0)

    image = pintar_bordes(image, bordes=caras, color=verde)
    image = pintar_bordes(image, bordes=ojos1, color=rojo)
    image = pintar_bordes(image, bordes=sonrisas1, color=azul)

    return image


def main():
    image = Img.cargar_imagen("../img/suaves-foto.jpg")
    # Camera.video_capture(detectar_cosas)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = detectar_cosas(image)
    Img.mostrar(image)

    Camera.video_capture(detectar_cosas)


if __name__ == '__main__':
    main()
