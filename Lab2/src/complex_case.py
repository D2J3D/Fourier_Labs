import numpy as np
import math
from numpy import sin, pi, exp, cos
import matplotlib.pyplot as plt


def original_function(t, a, b, c):
    return a * exp(-b * abs(t + c))



if __name__ == "__main__":
    a, b = 2, 4
    coeff = [-3, 1, 10]
    for c in coeff:
        fig, ax = plt.subplots(2,2)
        x = np.linspace(-15, 15, 1000)
        fourier_image_function = lambda t: np.exp(1j * t * c) * np.sqrt(2 / pi) * ((1/(4 - 1j * t)) * exp(-c * (4 - 1j * t)) - (1/(4 + 1j * t)) * exp(c * (4 + 1j*t)))
        y = [original_function(i, a, b, c) for i in x]
        comp_image = [fourier_image_function(i) for i in x]
        image_real_part = [np.real(i) for i in comp_image]
        image_complex_part = [np.imag(i) for i in comp_image]
        comp_image_moduled = [np.abs(i) for i in comp_image]
        

        ax[0,0].plot(x, y, label="g(t)")
        ax[0,1].plot(x,  image_real_part, label="Re(Fg(t))")
        ax[1,0].plot(x, image_complex_part,  label='Im(Fg(t))')
        ax[1, 1].plot(x, comp_image_moduled, label='|F(g(t))|')
        for i in range(0, 2):
            for j in range(2):
                ax[i, j].legend()
                ax[i, j].grid(True)
        plt.show()
        # plt.savefig("./image/complex_case_img/complex_" + str(c) + ".jpg")
