from utils import Img
import cv2 as cv


def main():
    image = Img.cargar_imagen("img/chica.png")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    Img.separar_rgb(image)

    Img.mostar_histograma(image)
    eq1, eq2, eq3 = Img.histograma_equalized(image)

    Img.show_3_gray(eq1, eq2, eq3)

    final_eq = Img.unir_colores(eq1, eq2, eq3)
    Img.mostrar_rgb(final_eq)


if __name__ == '__main__':
    main()
