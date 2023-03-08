from utils import Camera, Img
import mediapipe as mp
import numpy as np


def pintar_partes_cuerpo(frame: np.ndarray, params):
    pose = params["pose"]
    mp_pose = params["mp_pose"]
    mp_drawing = params["mp_drawing"]
    results = pose.process(image=frame)
    pose_landmarks = results.pose_landmarks
    if not pose_landmarks:
        return frame
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    mazo_zurda = pose_landmarks.landmark[16]
    width = frame.shape[1]
    height = frame.shape[0]

    x = round(mazo_zurda.x * width)
    y = round(mazo_zurda.y * height)
    z = round(mazo_zurda.z, 2)
    string_x = f"x: {x}"
    string_y = f"y: {y}"
    string_z = f"z: {z}"
    Img.escribir(frame, pos=(int(x), int(y)), size=4, text=string_x)
    Img.escribir(frame, pos=(int(x), int(y) + 20), size=4, text=string_y)
    Img.escribir(frame, pos=(int(x), int(y) + 40), size=4, text=string_z)

    return frame


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.7
    ) as pose:
        Camera.video_capture(
            operacion=pintar_partes_cuerpo,
            operacion_params={
                "pose": pose,
                "mp_pose": mp_pose,
                "mp_drawing": mp_drawing
            }
        )


if __name__ == '__main__':
    main()
