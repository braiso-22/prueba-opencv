import cv2 as cv
import numpy as np
from utils import Camera


def tapar_fondo(frame: np.ndarray, img_fondo):
    cv.accumulateWeighted(frame, img_fondo, 0.22)
    img_ok = cv.convertScaleAbs(img_fondo)
    dif = cv.absdiff(frame, img_ok)
    dif = cv.cvtColor(dif, cv.COLOR_BGR2GRAY)
    nada, mascara = cv.threshold(dif, 20, 255, cv.THRESH_BINARY)
    mascara = cv.morphologyEx(
        mascara,
        cv.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=7
    )
    final = cv.bitwise_and(frame, frame, mask=mascara)
    return final


def tapar_fondo_bgsb(frame: np.ndarray, fgbg):
    mascara = fgbg.apply(frame)
    mascara = cv.morphologyEx(
        mascara,
        cv.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=5
    )
    final = cv.bitwise_and(frame, frame, mask=mascara)
    return final


def main():
    captura_video = cv.VideoCapture(1, cv.CAP_DSHOW)
    cv.namedWindow("Video")
    ret, img_fondo = captura_video.read()
    img_fondo = np.float32(img_fondo)
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    accion = 1
    while True:
        frame = Camera.obtener_frame(captura_video)
        # final = tapar_fondo(frame, img_fondo)
        final = tapar_fondo_bgsb(frame, fgbg)
        accion = Camera.mostrar_frame(final, accion)
        if accion == -1:
            break
        if accion == 0:
            accion = 0
        else:
            accion = 1


if __name__ == '__main__':
    main()
