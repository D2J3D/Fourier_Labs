import math
from numpy import exp, sin, cos, sqrt, pi, inf
import numpy as np
import matplotlib.pyplot as plt

def integrator(function, a, b):
    step = 0.001
    t = np.linspace(a, b, int((b - a) / step))
    return sum([function(i) * step for i in t])


def fitter(array):
    for i in range(len(array)):
        if abs(array[i]) < math.e**(-10):
            array[i] = 0
        array[i] = round(array[i], 2)
    return array    


def find_c_k(function, w_n, dt):
    c_k_real = integrator(lambda t: cos(w_n * t) * function(t), dt[0], dt[1])
    c_k_im = 1 * integrator(lambda t: sin(w_n * t) * function(t), dt[0], dt[1])
    return [c_k_real, c_k_im]

def get_c_k(function, T, dt, n):
    w_n = (2 * pi)/T
    c_coeffs_real = [0 for i in range(2 * n + 1)]
    c_coeffs_im = [0 for i in range(2 * n + 1)]
    for i in range(0, n + 1):
        c_k = find_c_k(function, 2 * pi * (-n+i) / T, dt)
        c_coeffs_real[i] = 1 / T * c_k[0]
        c_coeffs_im[i] = -1 / T * c_k[1]
        if (len(c_coeffs_real) - i - 1 < len(c_coeffs_real)):
            c_coeffs_real[len(c_coeffs_real) - i - 1] = 1 / T * c_k[0]
            c_coeffs_im[len(c_coeffs_im) - i - 1] = 1 / T *  c_k[1]
    c_coeffs_real[0] /= 2
    c_coeffs_real = fitter(c_coeffs_real)
    c_coeffs_im = fitter(c_coeffs_im)
    # all_complex_coeffs = [complex(c_coeffs_real[j], c_coeffs_im[j]) for j in range(len(c_coeffs_im))]
    return [c_coeffs_real, c_coeffs_im]



def create_fourier_exponential(function, T, dt, n):
    c_coeffs_comp = get_c_k(function, T, dt, n)
    w_n = (2 * pi)/T
    c_coeffs_real = c_coeffs_comp[0]
    c_coeffs_im = c_coeffs_comp[1]
    approx_real = lambda t: sum([(cos(w_n * (-n + i) * t) * c_coeffs_real[i] - sin(w_n * (-n + i) * t) * c_coeffs_im[i]) for i in range(0, 2 * n + 1)])
    approx_im = lambda t: sum([(cos(w_n * (-n + i) * t) * c_coeffs_im[i] + c_coeffs_real[i] * sin(w_n * (-n + i) * t)) for i in range(0, 2 * n + 1)])
    function_approximation = lambda t: complex(approx_real(t), approx_im(t))
    return function_approximation


def original_function(t): 
    T, R = 8, 5 
    if -T / 8 <= t < T / 8:
        return complex(R, (8 * R * t) / T)
    elif T / 8 <= t < 3 * T / 8: 
       return complex(2 * R - (8 * R * t) / T, R)
    elif 3 * T / 8 <= t < 5 * T / 8:
        return complex(-R, 4 * R - (8 * R * t) / T)
    elif 5 * T / 8 <= t <= 7 * T / 8:
        return complex(-6 * R + (8 * R * t) / T, -R)

        
def save_approx(N):
    fig, ax = plt.subplots()
    T = 8
    R = 5
    x_1 = np.linspace(-T/8, 7*T/8, 100)
    y_1 = [original_function(i) for i in x_1] 
    fourier_approx_exp = create_fourier_exponential(original_function, T, [-T/8, 7*T/8], N)
    real_part = [np.real(original_function(i)) for i in x_1]
    complex_part = [np.imag(original_function(i)) for i in x_1]
    ax.plot(real_part, complex_part, label='original')
    y2 = [fourier_approx_exp(x_1[i]) for i in range(len(x_1))]
    real_approx = [np.real(i) for i in y2]
    complex_approx = [np.imag(i) for i in y2]
    ax.plot(real_approx, complex_approx, label='G_N')
    ax.grid(True)
    ax.legend()
    plt.savefig('G_' + str(N) + '_complex-func.jpg')


def save_Re(N):
    fig, ax = plt.subplots()
    T = 8
    R = 5
    x_1 = np.linspace(-T/8, 7*T/8, 100)
    y_1 = [np.real(original_function(i)) for i in x_1] 
    fourier_approx_exp = create_fourier_exponential(original_function, T, [-T/8, 7*T/8], N)
    real_part = [np.real(original_function(i)) for i in x_1]
    ax.plot(x_1, y_1)
    ax.plot(x_1, real_part, label='Re(f(t))')
    y2 = [fourier_approx_exp(x_1[i]) for i in range(len(x_1))]
    real_approx = [np.real(i) for i in y2]
    ax.plot(x_1, real_approx, label='Re(G_N)')
    ax.grid(True)
    ax.legend()
    plt.savefig('G_' + str(N) + '_complex-func_real_part.jpg')


def save_Im(N):
    fig, ax = plt.subplots()
    T = 8
    R = 5
    x_1 = np.linspace(-T/8, 7*T/8, 100)
    y_1 = [np.real(original_function(i)) for i in x_1] 
    fourier_approx_exp = create_fourier_exponential(original_function, T, [-T/8, 7*T/8], N)
    complex_part = [np.imag(original_function(i)) for i in x_1]
    ax.plot(x_1, complex_part, label='Im f(t)')
    y2 = [fourier_approx_exp(x_1[i]) for i in range(len(x_1))]
    complex_approx = [np.imag(i) for i in y2]
    ax.plot(x_1, complex_approx, label='Im G_N')
    ax.grid(True)
    ax.legend()
    plt.savefig('G_' + str(N) + '_complex-func_imag_part.jpg')



if __name__ == "__main__":
    n_s = [1, 2, 3,5, 10]
    for i in range(len(n_s)):
        save_approx(n_s[i])
        save_Re(n_s[i])
        save_Im(n_s[i])

