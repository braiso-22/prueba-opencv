from utils import Img
import cv2 as cv


def difuminaciones(image):
    image_suavizada = Img.suavizar(image)
    image_blur = cv.blur(image, (5, 5))
    image_gauss = cv.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)
    image_median = cv.medianBlur(image, 5)
    Img.mostrar_varios_rgb(
        image,
        image_suavizada,
        image_blur,
        image_gauss,
        image_median
    )


def realzados(image):
    realzada = Img.realzar(image)
    image_bilateral = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    Img.mostrar_varios_rgb(image, realzada, image_bilateral)


def bordes(image):
    image_gris = Img.escala_grises(image)
    image_gauss = cv.GaussianBlur(image_gris, (5, 5), sigmaX=0, sigmaY=0)
    Img.mostrar_varios_rgb(image_gris, image_gauss)

    laplaciano = cv.Laplacian(image_gauss, cv.CV_64F)
    sobelx = cv.Sobel(image_gauss, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(image_gauss, cv.CV_64F, 0, 1, ksize=5)
    sobelxy = sobely + sobelx
    Img.show_3_gray(laplaciano, sobelx, sobely)
    Img.mostrar(sobelxy, gris=True)


def main():
    image = Img.cargar_imagen("../img/chica.png")
    difuminaciones(image)
    realzados(image)

    cristalera = Img.cargar_imagen("../img/cristalera.jpg")
    bordes(cristalera)


if __name__ == '__main__':
    main()
