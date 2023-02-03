import cv2 as cv
import face_recognition


def main():
    captura_video = cv.VideoCapture(1, cv.CAP_DSHOW)
    while True:
        ret, frame = captura_video.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        cv.imshow("Video", frame)
        k = cv.waitKey(1)
        if k == 27:
            break
        pass
    captura_video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
