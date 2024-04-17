import numpy as np
import math
import matplotlib.pyplot as plt
import random
from numpy import sin

def g(t):
    if -0.5 <= t <= 0.5:
        return 2
    else:
        return 0


def time(T, dt):
    return np.arange(-T/2, T/2, dt)


def u(b, c, d, T, dt):
    segment = time(T, dt)
    print(random.uniform(0, len(segment)))
    return  lambda t: g(t) + b * (random.uniform(0, len(segment)) - 0.5)  


if __name__ == "__main__":
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 100)
    y = [u(0.1, 0, 0, 10, 0.01)(i) for i in x]
    ax.plot(x, y)
    ax.grid(True)
    plt.show()
    # ax.legend()