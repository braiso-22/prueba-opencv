import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from utils import Img


def operaciones_bits_2_images(img, img2):
    img_gris = Img.escala_grises(img)
    img2_gris = Img.escala_grises(img2)

    operacion_and = cv.bitwise_and(img_gris, img2_gris)

    operacion_or = cv.bitwise_or(img_gris, img2_gris)

    operacion_xor = cv.bitwise_xor(img_gris, img2_gris)
    operacion_not = cv.bitwise_not(img_gris)

    operacion_inverse_and = inverse_and(img_gris, img2_gris, pasar_a_gris=False)

    operacion_multiplicar_mascara = cv.bitwise_or(
        cv.multiply(operacion_and, 2),
        operacion_inverse_and
    )

    figura, ((eje1, eje2, eje3, eje4), (eje5, eje6, eje7, eje8)) = plt.subplots(nrows=2, ncols=4)
    eje1.imshow(img_gris, cmap="gray")
    eje2.imshow(img2_gris, cmap="gray")
    eje3.imshow(operacion_or, cmap="gray")
    eje4.imshow(operacion_and, cmap="gray")
    eje5.imshow(operacion_xor, cmap="gray")
    eje6.imshow(operacion_not, cmap="gray")
    eje7.imshow(operacion_inverse_and, cmap="gray")
    eje8.imshow(operacion_multiplicar_mascara, cmap="gray")
    plt.show()


def inverse_and(img, img2, pasar_a_gris=True):
    if pasar_a_gris:
        img = Img.escala_grises(img)
        img2 = Img.escala_grises(img2)
    operacion_and = cv.bitwise_and(img, cv.bitwise_not(img2))
    return operacion_and


def dibujar_lineas(img):
    # dibujar linea horizontal
    img[150] = 0

    # otra manera más controlada
    ''' 
    for i in range(img.shape[1]):
        img[152, i] = 0
        img[153, i] = [0, 255, 0]
    '''
    # dibujar linea vertical
    img[:, 150] = 0

    # otra manera más controlada
    '''
    for i in range(img.shape[0]):
        img[i, 152] = 0
        img[i, 153] = [0, 255, 0]
    '''


def pintar(chica):
    chica = cv.cvtColor(chica, cv.COLOR_BGR2RGB)
    dibujar_lineas(chica)
    Img.mostrar(chica)

    labios = chica[340:375, 250:350]
    # 0 = R, 1 = G, 2 = B
    canal = 1
    labios[:, :, canal] = 255
    Img.mostrar(chica)


def ver_propiedades_minmax(img):
    img = Img.escala_grises(img)
    minim, maxim, min_loc, max_loc = cv.minMaxLoc(img)
    print(f"min: {minim}, max: {maxim}, min_loc: {min_loc}, max_loc: {max_loc}")
    return minim, maxim, min_loc, max_loc


def ver_media_desviacion(img):
    img = Img.escala_grises(img)
    media, desviacion = cv.meanStdDev(img)
    print(f"media: {media}, desviacion: {desviacion}")
    return media, desviacion


def main():
    rectangulo = Img.cargar_imagen("../img/test5.png")
    circulo = Img.cargar_imagen("../img/test6.png")
    chica = Img.cargar_imagen("../img/chica.png")
    operaciones_bits_2_images(circulo, rectangulo)
    operaciones_bits_2_images(chica, circulo)
    Img.ver_cambios_entre(
        Img.cargar_imagen("../img/banda_caja1.png"),
        Img.cargar_imagen("../img/banda_caja2.png")
    )
    pintar(chica)
    chica = Img.cargar_imagen("../img/chica.png")
    ver_propiedades_minmax(chica)
    media, _ = ver_media_desviacion(chica)
    chica_gris = Img.escala_grises(chica)
    chica_bn = Img.binarizar(chica_gris, media)
    Img.mostrar(chica_bn, gris=True)
    normalizado = Img.normalizar_entre_01(chica)
    print("Normalizado")
    ver_propiedades_minmax(normalizado)
    ver_media_desviacion(normalizado)


if __name__ == '__main__':
    main()
