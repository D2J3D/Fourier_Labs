import math
from numpy import sin, cos, exp, pi
import numpy as np
import matplotlib.pyplot as plt

def integrator(function, a, b):
    # функция для численного интегрирования
    step = 0.001
    t = np.linspace(a, b, int((b - a) / step))
    return sum([function(i) * step for i in t])


def fitter(array):
    # обнуление слишком маленьких значений, и округление оставшихся
    for i in range(len(array)):
        if abs(array[i]) < math.e**(-10):
            array[i] = 0
        array[i] = round(array[i], 2)
    return array


def find_a_k(function, w_n, dt):
    # подсчет коэффициента a_k 1<=k<=N для ряда Фурье
    a_func = lambda t: function(t) * cos(w_n * t)
    a_k = integrator(a_func, dt[0], dt[1])
    if abs(a_k) < math.e**(-10):
        return 0.0
    return a_k


def find_b_k(function, w_n, dt):
    # подсчет коэффицента b_k 1 <= k <= N разложения для ряда Фурье 
    b_func = lambda t: function(t) * sin(w_n * t)
    b_k = integrator(b_func, dt[0], dt[1])
    if abs(b_k) < math.e**(-10):
        return 0.0
    return b_k


def find_c_k(function, w_n, dt):
    # подсчет коэффициента c_k для ряда Фурье в экспоненциальном виде
    c_k_real = integrator(lambda t: cos(w_n * t) * function(t), dt[0], dt[1])
    c_k_im = 1 * integrator(lambda t: sin(w_n * t) * function(t), dt[0], dt[1])
    return [c_k_real, c_k_im]


def get_c_k(function, T, dt, n):
    # подсчет коэффицентов ряда Фурье в exp форме
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
    return [c_coeffs_real, c_coeffs_im]


def create_fourier_exponential(function, T, dt, n):
    # разложение функции  в ее ряд Фурье в экспоненциальной форме
    c_coeffs_comp = get_c_k(function, T, dt, n)
    w_n = (2 * pi)/T
    c_coeffs_real = c_coeffs_comp[0]
    c_coeffs_im = c_coeffs_comp[1]
    approx_real = lambda t: sum([(cos(w_n * (-n + i) * t) * c_coeffs_real[i] - sin(w_n * (-n + i) * t) * c_coeffs_im[i]) for i in range(0, 2 * n + 1)])
    approx_im = lambda t: sum([(cos(w_n * (-n + i) * t) * c_coeffs_im[i] + c_coeffs_real[i] * sin(w_n * (-n + i) * t)) for i in range(0, 2 * n + 1)]) # actually, just a zero, because function is Real only
    function_approximation = lambda t: approx_real(t)
    return function_approximation


def create_fourier_classical(function, T, dt, n):
    # разложение функции в классический ряд Фурье 
    a_coeffs = [2 / T * find_a_k(function, 2 * pi * k/T, dt) for k in range(0, n+1)]
    a_coeffs[0] /= 2
    b_coeffs = [2 / T * find_b_k(function, 2 * pi * k/T, dt) for k in range(0, n+1)]
    approx = lambda t: sum([a_coeffs[i] * cos(2 * pi * i * t/T) + b_coeffs[i] * sin(2 * pi * i * t/T) for i in range(n)] )
    return approx


def get_coeffs_real(function, T, dt, n):
    # получение всех a_n ,b_n разложения в ряд Фурье
    a_coeffs = [2 / T * find_a_k(function, 2 * pi * k/T, dt) for k in range(0, n+1)]
    a_coeffs[0] /= 2
    b_coeffs = [2 / T * find_b_k(function, 2 * pi * k/T, dt) for k in range(0, n+1)]
    return a_coeffs, b_coeffs

def check_Parseval_real(function, T, dt, a_k, b_k):
    step = 0.00001
    a, b = dt[0], dt[1]
    a_k[0] *= 2 / np.sqrt(2)
    t = np.linspace(a, b, int((b - a) / step))
    left_part = 2 / T * sum([function(i) * function(i) * step for i in t])
    right_part = sum([a_k[i]**2 + b_k[i]**2 for i in range(len(b_k))])
    return left_part, right_part


def check_Parseval_exponential(function, T, dt, c_k):
    step = 0.00001
    a, b = dt[0], dt[1]
    t = np.linspace(a, b, int((b - a) / step))
    left_part = 2 / T * sum([function(i) * np.conjugate(function(i)) * step for i in t])
    right_part = sum([(c_k[0][i]**2 + c_k[1][i]**2) for i in range(len(c_k))])
    return left_part, right_part

def original_function(t):
    return t + t**3 + 2**t



def save_Fn(original_function, N):
    fig, ax = plt.subplots()
    x = np.linspace(1, 10, 100)
    y_1 = [original_function(i) for i in  x_1]
    fourier_approx = create_fourier_classical(original_function, 4, [-2, 2], N)
    y_2 = [fourier_approx(x_1[i]) for i in range(len(x))]
    ax.plot(x, y_1, label='original')
    ax.plot(x, y_2, label='F_n = ' + str(N))
    ax.grid(True)
    ax.legend()
    plt.savefig('F_' + str(N) + '_classical.jpg')


def save_Gn(original_function, N):
    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 100)
    y_1 = [original_function(i) for i in x_1]
    fourier_approx = create_fourier_exponential(original_function, 4, [-2, 2], N)
    y_2 = [fourier_approx(x_1[i]) for i in range(len(x))]
    ax.plot(x, y_1, label='original')
    ax.plot(x, y_2, label='G_n = ' + str(N))
    ax.grid(True)
    ax.legend()
    plt.savefig('G_' + str(N) + '_exponential.jpg')
    


if __name__ == "__main__":
    fig, ax = plt.subplots()
    x_1 = np.linspace(-2, 2, 100)
    y_1 = [original_function(i) for i in x_1]
    # all_coeffs_real = get_coeffs_real(original_function, 4, [-2 , 2], 5)
    all_coeffs_real = get_coeffs_real(original_function, 4, [-2, 2], 25)
    all_coeffs_exponential = get_c_k(original_function, 4, [-2, 2], 25)

    print("info ", check_Parseval_real(original_function, 4, [-2, 2], all_coeffs_real[0], all_coeffs_real[1]))
    print("info ", check_Parseval_exponential(create_fourier_exponential(original_function, 4, [-2, 2], 25), 4, [-2, 2], all_coeffs_exponential))
