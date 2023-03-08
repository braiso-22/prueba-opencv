import sys, getopt
import cv2 as cv
import numpy as np
from utils import Camera
from utils import Img

color_punto = None

img_bgr = None


def click_raton(event, x, y, flags, param):
    global img_bgr
    global color_punto
    if event == cv.EVENT_LBUTTONDBLCLK:
        if color_punto is None:
            b, g, r = img_bgr[y, x]
            color_punto = np.uint8([[[b, g, r]]])
            color_punto = cv.cvtColor(color_punto, cv.COLOR_BGR2HSV)
            print(color_punto)
        else:
            color_punto = None
            print("Desactivado")


def main():
    global img_bgr
    vid = cv.VideoCapture(0)
    cv.namedWindow("Objeto")
    cv.setMouseCallback("Objeto", click_raton)
    colores = [Img.ROJO, Img.VERDE, Img.AZUL]
    color_actual = 0
    centro_circulos = [(50, 50), (150, 50), (250, 50)]
    lista_puntos = []
    while True:
        ret, img_bgr = vid.read()
        img_bgr = cv.flip(img_bgr, 1)
        if color_punto is not None:
            img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
            for i in range(3):
                img_bgr = cv.circle(img_bgr, centro_circulos[i], 30, colores[i], -1)
            h = color_punto[0][0][0]
            haux = max(h - 5, 0)
            color_minimo = np.array([haux, 40, 30])
            haux = min(h + 10, 180)
            color_maximo = np.array([haux, 255, 255])
            mascara = cv.inRange(img_hsv, color_minimo, color_maximo)
            mascara = cv.erode(mascara, None, iterations=3)
            mascara = cv.dilate(mascara, None, iterations=4)
            (contornos, jerarquia) = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for contorno in contornos:
                area = cv.contourArea(contorno)
                if area > 3500:
                    cv.drawContours(img_bgr, [contorno], 0, (0, 255, 0), 2)
                    M = cv.moments(contorno)
                    if M["m00"] != 0:
                        x = int(M["m10"] / M["m00"])
                        y = int(M["m01"] / M["m00"])
                        for i, cc in enumerate(centro_circulos):
                            if abs(cc[0] - x) + abs(cc[1] - y) < 30:
                                color_actual = i
                                break
                        lista_puntos.append((x, y, color_actual))
                        cv.circle(img_bgr, (x, y), 7, (255, 255, 255), -1)
                        cv.putText(img_bgr, "centro" + f"x{x},y{y}", (x - 20, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (0, 0, 255), 2)
            cv.imshow("mascara", mascara)

            for punto in lista_puntos:
                x, y, n_color = punto
                if n_color == -1:
                    color = Img.TRANSPARENTE
                else:
                    color = colores[n_color]
                cv.circle(img_bgr, (x, y), 7, color, -1)
        cv.imshow("Objeto", img_bgr)

        if cv.waitKey(10) & 0xFF == 27: break

    vid.release()


if __name__ == '__main__':
    main()
