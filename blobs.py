from utils import Img
import cv2 as cv
import numpy as np


def param_cafe():
    param = cv.SimpleBlobDetector_Params()
    param.filterByArea = True
    param.minArea = 1000
    param.maxArea = 5000
    return param


def param_lentejas():
    param = cv.SimpleBlobDetector_Params()
    param.filterByArea = True
    param.minArea = 150
    param.maxArea = 500
    param.minConvexity = 97 / 100
    return param


def param_arroz():
    param = cv.SimpleBlobDetector_Params()
    param.filterByArea = True
    param.minArea = 150
    param.maxArea = 500
    param.filterByInertia = True
    param.minInertiaRatio = 0.1
    param.maxInertiaRatio = 0.7
    param.minConvexity = 0.5
    return param


def blobs(image: np.ndarray):
    # if image side id bigger than 500 resize it to the 25% of its size
    if image.shape[0] > 500:
        image = cv.resize(image, (0, 0), fx=0.25, fy=0.25)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    _, _, b = Img.separar_rgb(image)

    detector_cafe = cv.SimpleBlobDetector_create(param_cafe())
    detector_lentejas = cv.SimpleBlobDetector_create(param_lentejas())
    detector_arroz = cv.SimpleBlobDetector_create(param_arroz())
    keypoints_cafe = detector_cafe.detect(b)
    keypoints_lentejas = detector_lentejas.detect(b)
    keypoints_arroz = detector_arroz.detect(b)
    image2 = image.copy()
    image2 = cv.drawKeypoints(image2, keypoints_cafe, np.array([]), (255, 255, 0),
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv.drawKeypoints(image2, keypoints_lentejas, np.array([]), (0, 255, 255),
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv.drawKeypoints(image2, keypoints_arroz, np.array([]), (255, 0, 255),
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    Img.mostrar(image)
    Img.mostrar(image2)


def main():
    image = Img.cargar_imagen("img/granos.jpg")
    blobs(image)


if __name__ == '__main__':
    main()
