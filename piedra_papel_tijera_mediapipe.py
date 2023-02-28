import mediapipe as mp
import cv2 as cv
import tensorflow as tf
import numpy as np


def menu_juego(image, entrada):
    cv.putText(
        image,
        "Pulgar arriba para empezar, pulgar abajo para salir",
        (50, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )
    if entrada == "ok":
        return 1
    elif entrada == "no":
        return -1
    else:
        return None


def jugada_cpu():
    opciones = ["piedra", "papel", "tijera"]
    return opciones[np.random.randint(0, 3)]


def resultado_jugada(player, cpu):
    if player == cpu:
        return "Empate"
    elif player == "piedra" and cpu == "tijera":
        return "Gana jugador"
    elif player == "papel" and cpu == "piedra":
        return "Gana jugador"
    elif player == "tijera" and cpu == "papel":
        return "Gana jugador"
    else:
        return "Gana CPU"


def cargar_modelo(route: str):
    try:
        model = tf.keras.models.load_model(route, compile=False)
        return model
    except ImportError as importError:
        print("Error al cargar el modelo: ", importError)
        return None


def landmarks_to_screen_points(hand_landmarks, image_width, image_height):
    landmarks_in_screen = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmarks_in_screen.append((x, y))
    return landmarks_in_screen


def predecir_posicion_mano(modelo, lista_puntos, nombres_posiciones):
    prediccion = modelo.predict([lista_puntos])
    indice = np.argmax(prediccion)
    nombre = nombres_posiciones[indice]
    return nombre


def translate_posicion(posicion):
    traducciones = {
        "fist": "piedra",
        "stop": "papel",
        "live long": "papel",
        "peace": "tijera",
        "thumbs up": "ok",
        "thumbs down": "no",
        "call me": "",
        "okay": "ok",
        "rock": "",
        "smile": "",
    }
    return traducciones[posicion]


def is_valid_movement(movement):
    valid_movements = ["piedra", "papel", "tijera"]
    return movement in valid_movements


def procesar_frame(image, hands, mp_drawing, mp_hands, modelo, nombres_posiciones, estado, movimiento_anterior):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    posicion_actual = None
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_in_screen = landmarks_to_screen_points(hand_landmarks, image.shape[1], image.shape[0])
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            posicion_actual = predecir_posicion_mano(modelo, landmarks_in_screen, nombres_posiciones)

        if translate_posicion(posicion_actual) != "":
            posicion_actual = translate_posicion(posicion_actual)
        else:
            posicion_actual = movimiento_anterior
        image = cv.flip(image, 1)
        if estado == 0:
            estado = menu_juego(image, posicion_actual) if menu_juego(image, posicion_actual) else 0
        elif estado == 1:
            cv.putText(
                image,
                f"Jugada del jugador: {movimiento_anterior}. Pulgar arriba para jugar",
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
            if posicion_actual == "ok" and is_valid_movement(movimiento_anterior):
                estado = 2

                pass
    else:
        image = cv.flip(image, 1)

    return image, estado, posicion_actual


def capturar_video_mano():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)
    modelo = cargar_modelo(
        "./models/hands/mp_hand_gesture"
    )

    # open file
    with open("./models/hands/gesture.names", "r") as f:
        nombres_posiciones = f.read().splitlines()

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
    ) as hands:
        estado = 0
        movimiento_anterior = None
        while cap.isOpened():
            # Leer frame
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            # procesar frame
            image, estado, movimiento_anterior = procesar_frame(image, hands, mp_drawing, mp_hands, modelo,
                                                                nombres_posiciones, estado, movimiento_anterior)
            # mostrar frame
            cv.imshow('MediaPipe Hands', image)
            # salir
            if cv.waitKey(5) & 0xFF == 27 or estado == -1:
                break
    cap.release()


def main():
    capturar_video_mano()


if __name__ == '__main__':
    main()
