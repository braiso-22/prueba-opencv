from utils import Img
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    image = Img.cargar_imagen("img/diagonal_negra.png")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    Img.mostrar(image, gris=True)
    image_bin2 = cv.threshold(image, 200, 255, cv.THRESH_BINARY)[1]
    image_bin3 = cv.threshold(image, 200, 255, cv.THRESH_BINARY_INV)[1]
    image_bin4 = cv.threshold(image, 127, 255, cv.THRESH_TRUNC)[1]
    image_bin5 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO)[1]
    image_bin6 = cv.threshold(image, 0, 100, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    fig, (ax) = plt.subplots(1, 6)
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(image_bin2, cmap="gray")
    ax[2].imshow(image_bin3, cmap="gray")
    ax[3].imshow(image_bin4, cmap="gray")
    ax[4].imshow(image_bin5, cmap="gray")
    ax[5].imshow(image_bin6, cmap="gray")
    plt.show()
    pass


if __name__ == '__main__':
    main()
