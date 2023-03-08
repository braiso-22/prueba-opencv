from utils import Camera
import cv2 as cv
import numpy as np


def click_raton(event, x, y, flags, param):
    image = param

    if event == cv.EVENT_LBUTTONDBLCLK:
        b, g, r = image[y, x]
        color_punto = np.uint8([[[b, g, r]]])
        color_punto = cv.cvtColor(color_punto, cv.COLOR_BGR2HSV)
        hsup = max(color_punto[0][0][0] - 10, 0)
        hinf = min(color_punto[0][0][0] + 10, 180)
        color_min = np.array([hsup, 20, 20])
        color_max = np.array([hinf, 255, 255])
        mascara = cv.inRange(image, color_min, color_max)
        cv.imshow("Mascara", mascara)

        print(f"Color del punto(x:{x},y{y})", color_punto)


def main():
    Camera.video_capture(mouse_callback=click_raton)


if __name__ == '__main__':
    main()
