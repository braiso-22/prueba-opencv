import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = 0.1 * np.sin(40 * x)
    y3 = y1 + y2
    plt.plot(x, y1 + 2, color="blue")
    plt.plot(x, y2 + 2, color="green")
    plt.plot(x, y3 + 2, color="red")

    f1 = np.ones(11)
    f1 = f1 / f1.sum()
    print(f1)
    r1 = []
    i = 0
    while i < (len(x) - len(f1)):
        s1 = y3[i:i + len(f1)]
        s2 = (s1 * f1).sum()
        r1.append(s2)
        i += 1
    plt.plot(x[len(f1) // 2:-len(f1) // 2], r1, color="orange")
    plt.show()


if __name__ == '__main__':
    main()
