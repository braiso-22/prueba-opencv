import numpy as np
from utils import Img
import cv2 as cv


def obtener_mascara_matricula(img: np.ndarray):
    matricula_gris = Img.escala_grises(img)
    matricula_desenfocada = cv.GaussianBlur(matricula_gris, (7, 7), 0)
    _, threshold = cv.threshold(matricula_desenfocada, 127, 255, cv.THRESH_BINARY_INV)

    return threshold


def pintar_bordes_matricula(img: np.ndarray, mascara_threshold):
    analisis = cv.connectedComponentsWithStats(mascara_threshold, 4)
    total_labels, labels, stats, centroids = analisis
    output = np.zeros(Img.escala_grises(img).shape, np.uint8)
    for i in range(1, total_labels):
        mascara = (labels == i).astype(np.uint8) * 255
        output = cv.bitwise_or(output, mascara)
        area = stats[i, cv.CC_STAT_AREA]
        punto_sup_izq = (stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP])
        punto_inf_der = (stats[i, cv.CC_STAT_LEFT] + stats[i, cv.CC_STAT_WIDTH],
                         stats[i, cv.CC_STAT_TOP] + stats[i, cv.CC_STAT_HEIGHT])
        (x, y) = centroids[i]
        cv.rectangle(img, punto_sup_izq, punto_inf_der, (0, 255, 0), 2)
        cv.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)


def main():
    matricula = Img.cargar_imagen("img/matricula.jpg")
    mascara = obtener_mascara_matricula(matricula)
    pintar_bordes_matricula(matricula, mascara)
    Img.mostrar(matricula, gris=True)


if __name__ == '__main__':
    main()
