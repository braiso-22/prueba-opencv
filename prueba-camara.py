import cv2 as cv
import face_recognition
import numpy as np


def obtener_frame(capturadora: cv.VideoCapture):
    ret, frame = capturadora.read()
    if not ret:
        exit(0)

    frame = cv.flip(frame, 1)
    return frame


def pintar_borde_cara(frame: np.ndarray):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv.rectangle(frame, (left, top), (right, bottom), (50, 200, 50), 2)


def mostrar_frame(frame: np.ndarray, accion: int):
    cv.imshow("Video", frame)
    k = cv.waitKey(accion)
    if k == ord(' ') & 0xFF:
        if accion == 1:
            return 0
        else:
            return 1
    if k == ord('q') & 0xFF:
        return -1


def main():
    captura_video = cv.VideoCapture(1, cv.CAP_DSHOW)
    accion = 1
    while True:
        frame = obtener_frame(captura_video)
        modo = "Pausa" if accion == 0 else "Reproduciendo"
        ayuda = "(espacio para cambiar, q para salir)"
        cv.putText(frame, modo, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, ayuda, (250, 450), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        pintar_borde_cara(frame)

        accion = mostrar_frame(frame, accion)
        if accion == -1:
            break
        if accion == 0:
            accion = 0
        else:
            accion = 1

    captura_video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
