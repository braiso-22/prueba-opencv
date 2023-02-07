from utils import Img
import cv2 as cv
import matplotlib.pyplot as plt


def binarizar(image):
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


def binarizar2(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.medianBlur(image, 5)

    # Global thresholding
    ret, th1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    # Adaptive thresholding
    th2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    ret, th4 = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    # Plotting the images using matplotlib
    titles = ['Original', 'Global Thresholding',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu']
    images = [image, th1, th2, th3, th4]
    # Specifying the grid size
    plt.figure(figsize=(10, 20))
    # Number of images in the grid 2*2 = 4
    for i in range(5):
        plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    # Displaying the grid
    plt.show()


def main():
    image = Img.cargar_imagen("img/diagonal_negra.png")
    binarizar(image)
    image2 = Img.cargar_imagen("img/sudoku.png")
    binarizar2(image2)
    pass


if __name__ == '__main__':
    main()
