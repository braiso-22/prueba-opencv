import mediapipe as mp
import cv2 as cv
import tensorflow as tf
import numpy as np


class Juego:
    def __init__(self):
        self.opciones = ["piedra", "papel", "tijera"]
        self.estado = 0
        self.jugada_cpu = self.opciones[np.random.randint(0, 3)]
        self.jugada_jugador = None
        pass

    def mostrar_menu_juego(self, image):
        cv.putText(
            image,
            "Pulgar arriba para empezar, pulgar abajo para salir",
            (50, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv.LINE_AA
        )

    def opcion_menu_juego(self, entrada):
        if entrada == "ok":
            return 1
        elif entrada == "no":
            return -1
        else:
            return None

    def resultado_jugada(self, player: str):
        if player == self.jugada_cpu:
            return "Empate"
        elif player == "piedra" and self.jugada_cpu == "tijera":
            return "Gana jugador"
        elif player == "papel" and self.jugada_cpu == "piedra":
            return "Gana jugador"
        elif player == "tijera" and self.jugada_cpu == "papel":
            return "Gana jugador"
        else:
            return "Gana CPU"

    def __rectangle__(self, image, text_len):
        cv.rectangle(
            image,
            (50, 40),
            (text_len * 10, 60),
            (0, 0, 0),
            cv.FILLED
        )

    def mostrar_menu_actual(self, image):
        if self.estado == 0:
            self.mostrar_menu_juego(image)
        elif self.estado == 1:
            string_a_mostrar = f"Jugada del jugador: {self.jugada_jugador}. Pulgar arriba para jugar"
            self.__rectangle__(image, len(string_a_mostrar))
            cv.putText(
                image,
                string_a_mostrar,
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
        elif self.estado == 2:
            string_a_mostrar = f"Jugada del jugador: {self.jugada_jugador}. Jugada de la CPU: {self.jugada_cpu}"
            self.__rectangle__(image, len(string_a_mostrar))
            cv.putText(
                image,
                string_a_mostrar,
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
            cv.putText(
                image,
                f"{self.resultado_jugada(self.jugada_jugador)}",
                (50, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
            string_a_mostrar = "Pulgar arriba para continuar"
            cv.putText(
                image,
                string_a_mostrar,
                (50, 150),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
        elif self.estado == 3:
            string_a_mostrar = "Pulgar arriba para volver a jugar, pulgar abajo para salir"
            self.__rectangle__(image, len(string_a_mostrar))
            cv.putText(
                image,
                string_a_mostrar,
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

    def cambiar_estado_juego(self, posicion_actual):
        if self.estado == 0:
            self.estado = self.opcion_menu_juego(posicion_actual) if self.opcion_menu_juego(posicion_actual) else 0
        elif self.estado == 1:
            if posicion_actual == "ok" and is_valid_movement(self.jugada_jugador):
                self.estado = 2
            else:
                self.jugada_jugador = posicion_actual
        elif self.estado == 2:
            if posicion_actual == "ok":
                self.estado = 3
        elif self.estado == 3:
            if posicion_actual == "ok":
                self.estado = 1
                self.jugada_cpu = self.opciones[np.random.randint(0, 3)]
            elif posicion_actual == "no":
                self.estado = -1


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


def procesar_frame(image, hands, mp_drawing, mp_hands, modelo, nombres_posiciones, juego, num_frames):
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
            posicion_sin_traducir = predecir_posicion_mano(modelo, landmarks_in_screen, nombres_posiciones)
            posicion_actual = translate_posicion(posicion_sin_traducir)
        image = cv.flip(image, 1)
        juego.mostrar_menu_actual(image)
        if num_frames % 30 == 0:
            juego.cambiar_estado_juego(posicion_actual)

    else:
        image = cv.flip(image, 1)

    return image, posicion_actual


def capturar_video_mano():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(0)
    modelo = cargar_modelo(
        "../models/hands/mp_hand_gesture"
    )

    # open file
    with open("../models/hands/gesture.names", "r") as f:
        nombres_posiciones = f.read().splitlines()

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
    ) as hands:
        juego = Juego()
        frame_num = 0
        while cap.isOpened():
            # Leer frame
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            # procesar frame
            frame_num += 1
            image, movimiento_anterior = procesar_frame(image, hands, mp_drawing, mp_hands, modelo,
                                                        nombres_posiciones, juego, frame_num)
            # mostrar frame
            cv.imshow('MediaPipe Hands', image)
            # salir
            if cv.waitKey(5) & 0xFF == 27 or juego.estado == -1:
                break
    cap.release()


def main():
    capturar_video_mano()


if __name__ == '__main__':
    main()
