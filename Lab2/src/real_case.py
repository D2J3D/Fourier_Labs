import numpy as np
import math
import matplotlib.pyplot as plt


def original_function(t, a, b):
    return a * np.exp(-b * t**2)


def sq_wave_fourier_image(t, a, b):
    if (abs(t) <= b):
        return a / (b * np.sqrt(2 * math.pi))
    else:
        return 0

if __name__ == "__main__":
    coeffs = [[1, 2], [2, 3], [3, 4], [1, 1], [0.5, 1]]
    for j in range(len(coeffs)):
        fig, ax = plt.subplots()
        a, b = coeffs[j][0], coeffs[j][1]
        x = np.linspace(-15, 15, 1000)
        fourier_image_function = lambda t: a/(np.sqrt(2)*b)* np.exp(-t**2/(4*b))
        y = [original_function(i, a, b) for i in x]
        image = [fourier_image_function(i) for i in x]
        plt.grid(True)
        ax.plot(x, y, label='original')
        ax.plot(x, image, label='image')
        ax.legend()
        plt.savefig(f"image/gauss_function_{a}_{b}.jpg")
