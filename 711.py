import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import lambertw

# common for all constants
t_step = 0.2
c_step = 1

# dN/dt =  r * N ^ 2
# it's solution is function
# N(t) = 1.0 / (c1 - r * t)
# c1 and r are consts
r = 2.13


def function1(c1, t, rl):
    return 1.0 / (c1 - rl * t)


def draw_function_1(c_min, c_max, t_min, t_max, r):
    plt.subplot(2, 1, 2)
    plt.title('dN/dt =  r * N ^ 2 solutions '
              '\n  C = [{0} : {1}] , t = [{2} : {3}] , r = {4}'.format(c_min, c_max, t_min, t_max, r))
    plt.xlabel('Time')
    plt.ylabel('Popultaion size')
    for c1 in np.arange(c_min, c_max, c_step):
        t_arr = np.arange(t_min, t_max, t_step)
        n_arr = [function1(c1, t, r) for t in t_arr]
        plt.plot(t_arr, n_arr)


draw_function_1(5, 10, 0, 10, r)

# dN/dt =  a * ((b * N ^ 2) /(b  + v * N))
# it's solution is function
# N(t) = b /(v *  W(b/v * e^(c_1 - (b * t) / a)))
# a,b,v  are consts and > 0

a = 1
b = 2
v = 3


def function2(a, b, v, c1, t):
    W = (b / v) * math.e ** (c1 - t * (b / a))
    return b / (v * lambertw(W))


def draw_function_2(c_min, c_max, t_min, t_max, a, b, v):
    plt.subplot(2, 1, 1)
    plt.title('dN/dt =  a * ((b * N ^ 2) /(b  + v * N)) '
              '\n  C = [{0} : {1}] , t = [{2} : {3}] , a = {4}, b = {5}, v = {6}'.format(c_min, c_max, t_min, t_max,
                                                                                         a,
                                                                                         b, v))
    plt.ylabel('Popultaion size')
    for c1 in np.arange(c_min, c_max, c_step):
        t_arr = np.arange(t_min, t_max, t_step)
        n_arr = [function2(a, b, v, c1, t) for t in t_arr]
        plt.plot(t_arr, n_arr)


draw_function_2(5, 10, 0, 10, a, b, v)
plt.show()
