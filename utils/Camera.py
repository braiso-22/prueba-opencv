import numpy as np
import cv2 as cv
import face_recognition


def obtener_frame(capturadora: cv.VideoCapture):
    ret, frame = capturadora.read()
    if not ret:
        capturadora.release()
        cv.destroyAllWindows()
        exit(0)

    frame = cv.flip(frame, 1)
    return frame


def pintar_ayuda(frame: np.ndarray, accion: int):
    modo = "Pausa" if accion == 0 else "Reproduciendo"
    ayuda = "(espacio para cambiar, q para salir)"
    cv.putText(frame, modo, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, ayuda, (250, 450), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)


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


def pintar_borde_cara(frame: np.ndarray):
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv.rectangle(frame, (left, top), (right, bottom), (50, 200, 50), 2)


def get_camera():
    return cv.VideoCapture(1, cv.CAP_DSHOW)


def video_capture(operacion):
    captura_video = get_camera()
    cv.namedWindow("Video")
    accion = 1
    while True:
        frame = obtener_frame(captura_video)
        # funcion que se le pasa como parametro
        frame = operacion(frame)

        accion = mostrar_frame(frame, accion)
        if accion == -1:
            break
        if accion == 0:
            accion = 0
        else:
            accion = 1
