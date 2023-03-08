import mediapipe as mp
import cv2 as cv
from tensorflow.keras.models import load_model


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv.VideoCapture(1)
    with mp_hands.Hands(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(5) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    main()
