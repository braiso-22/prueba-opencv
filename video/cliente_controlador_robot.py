import socket

import mediapipe as mp
import cv2 as cv


def translate_posicion(posicion):
    traducciones = {
        "fist": "P",
        "stop": "L",
        "live long": "L",
        "peace": "J",
        "thumbs up": "W",
        "thumbs down": "down",
        "call me": "W",
        "okay": "ok",
        "rock": "down",
        "smile": "W",
    }
    return traducciones[posicion]


def get_position(punto_pulgar, punto_indice):
    posiciones = {
        "izquierda": "a",
        "derecha": "d",
        "arriba": "w",
        "abajo": "s",
    }
    if punto_pulgar[0] < punto_indice[0] and punto_pulgar[1] < punto_indice[1]:
        return posiciones["izquierda"]
    elif punto_pulgar[0] > punto_indice[0] and punto_pulgar[1] < punto_indice[1]:
        return posiciones["derecha"]
    elif punto_pulgar[0] < punto_indice[0] and punto_pulgar[1] > punto_indice[1]:
        return posiciones["arriba"]
    elif punto_pulgar[0] > punto_indice[0] and punto_pulgar[1] > punto_indice[1]:
        return posiciones["abajo"]
    pass


def procesar_frame(image, hands, mp_hands, operador: socket, num_frames):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]
            y_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0]
            x_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]
            y_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]

            posicion = get_position((x_pulgar, y_pulgar), (x_indice, y_indice))
        image = cv.flip(image, 1)
        arr = bytes(posicion, 'ascii')

        if num_frames % 23 == 0:
            operador.sendall(arr)
    else:
        image = cv.flip(image, 1)

    return image


def capturar_video_mano():
    input("recuerda iniciar el servidor pulsando espacio en webots antes de iniciar el cliente")
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
    ) as hands:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            HOST = "127.0.0.1"  # The server's hostname or IP address
            PORT = 50500  # The port used by the server
            server_address = (HOST, PORT)
            s.connect(server_address)

            frame_num = 0
            while cap.isOpened():
                # Leer frame
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                # procesar frame
                frame_num += 1
                image = procesar_frame(image, hands, mp_hands, s, frame_num)
                # mostrar frame
                cv.imshow('MediaPipe Hands', image)
                # salir
                if cv.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


def main():
    capturar_video_mano()


if __name__ == '__main__':
    main()
