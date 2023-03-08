from utils import Img
import cv2 as cv
import numpy as np
import math


def obtener_mascara_monedas(image: np.ndarray) -> np.ndarray:
    r, g, b = Img.separar_rgb(image)
    h, s, v = Img.separar_hsv(image)
    l, u, v = Img.separar_luv(image)
    # Img.mostrar_varios_rgb(b, s, u)
    kernel = np.ones((3, 3), np.uint8)
    b2 = cv.GaussianBlur(b, (5, 5), 0)

    binarizado_b = Img.binarizar(b2, 128)
    binarizado_b = cv.GaussianBlur(binarizado_b, (5, 5), 0)
    # Img.mostrar(binarizado_b, gris=True)
    binarizado_b = cv.morphologyEx(binarizado_b, cv.MORPH_CLOSE, kernel, iterations=4)
    binarizado_b = cv.morphologyEx(binarizado_b, cv.MORPH_OPEN, kernel, iterations=6)

    binarizado_u = Img.binarizar(u, 102)
    binarizado_u = cv.GaussianBlur(binarizado_u, (5, 5), 0)
    # Img.mostrar(binarizado_u, gris=True)
    binarizado_u = cv.morphologyEx(binarizado_u, cv.MORPH_CLOSE, kernel, iterations=2)
    binarizado_u = cv.morphologyEx(binarizado_u, cv.MORPH_OPEN, kernel, iterations=20)

    suma = cv.bitwise_and(binarizado_b, binarizado_u)
    suma = cv.GaussianBlur(suma, (3, 3), 0)
    suma = Img.binarizar(suma, 130)
    # Img.mostrar(suma, gris=True)
    suma = cv.morphologyEx(suma, cv.MORPH_CLOSE, kernel, iterations=5)
    suma = cv.morphologyEx(suma, cv.MORPH_OPEN, kernel, iterations=15)

    # Img.mostrar_varios_rgb(binarizado_b, binarizado_u, suma)

    suma = cv.bitwise_not(suma)
    # suma = cv.GaussianBlur(suma, (31, 31), 0)
    suma = cv.erode(suma, kernel, iterations=8)
    suma = cv.dilate(suma, kernel, iterations=10)
    suma = cv.morphologyEx(suma, cv.MORPH_CLOSE, kernel, iterations=20)
    # Img.mostrar(suma, gris=True)
    return suma


def obtener_canny(mascara: np.ndarray, umbral_min=50, umbral_max=100) -> np.ndarray:
    canny: np.ndarray = cv.Canny(mascara, umbral_min, umbral_max)
    # Img.mostrar(canny, gris=True)
    return canny


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


def pintar_sobre_monedas(image: np.ndarray, contornos: list) -> np.ndarray:
    monedas_rodeadas = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)

    # dibujar todos los contornos
    monedas_rodeadas = cv.drawContours(monedas_rodeadas, contornos, -1, (255, 0, 0), 2)

    perimetros = [cv.arcLength(contorno, True) for contorno in contornos]
    areas = [cv.contourArea(contorno) for contorno in contornos]
    centros_momento = [cv.moments(contorno) for contorno in contornos]
    area_maxima = max(areas)
    centros_puntos = []
    radios = []
    valores = []

    for i in range(len(contornos)):
        valor_moneda = set_valor(area_maxima, areas[i])
        valores.append(valor_moneda)
        centro = (
            int(centros_momento[i]['m10'] / centros_momento[i]['m00']),
            int(centros_momento[i]['m01'] / centros_momento[i]['m00'])
        )
        centros_puntos.append(centro)
        radio = int(perimetros[i] / (2 * math.pi))
        radios.append(radio)

    for i in range(len(contornos)):
        if areas[i] < area_maxima / 4:
            continue
        monedas_rodeadas = cv.circle(
            img=monedas_rodeadas,
            center=centros_puntos[i],
            radius=radios[i],
            color=(0, 255, 0),
            thickness=2,
        )
        elipse = cv.fitEllipse(contornos[i])
        cv.ellipse(monedas_rodeadas, elipse, (0, 0, 255), 2)
        cv.putText(
            img=monedas_rodeadas,
            text=(str(i)),
            org=centros_puntos[i],
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv.LINE_AA
        )
        cv.putText(
            img=monedas_rodeadas,
            text=("* Area: " + str(areas[i])),
            org=(centros_puntos[i][0] - radios[i] - 10, centros_puntos[i][1] - radios[i] - 10),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv.LINE_AA
        )
        # Solo si la imagen está escalada para que las monedas se vean sin deformación
        '''cv.putText(img=monedas_rodeadas,
                   text=str(valores[i]),
                   org=(centros_puntos[i][0] - radios[i] - 10, centros_puntos[i][1] + radios[i] + 10),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1,
                   color=(255, 0, 0),
                   thickness=2,
                   lineType=cv.LINE_AA
                   )'''
    return monedas_rodeadas


def main():
    image = Img.cargar_imagen("../img/monedas.png")
    mascara_monedas = obtener_mascara_monedas(image)
    canny = obtener_canny(mascara_monedas)

    contornos, jerarquia = cv.findContours(canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("Monedas encontradas: {}".format(len(contornos)))

    monedas_rodeadas = pintar_sobre_monedas(image, contornos)
    Img.mostrar(monedas_rodeadas)


if __name__ == '__main__':
    main()
