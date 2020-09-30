import matplotlib.pyplot as plt
import numpy as np
import math

def gaussion(x, mean, std=1):
    y = np.exp(-(x - mean)**2) / (np.sqrt(2*np.pi))
    return y

def kl_div(p1, p2):
    return p1*np.log(p1 / (p2+1e-8))

def js_div(p1, p2):
    return kl_div(p1, (p1 + p2)/2)/2. + kl_div(p2, (p1 + p2)/2)/2.

def plot(x, y):
    plt.figure()
    plt.xlim(-10, 40)
    plt.plot(x, y)
    plt.savefig('x_y.jpg')

x = np.linspace(-10, 40, 200)
y = gaussion(x, 0)
y1 = gaussion(x, 20)
js = js_div(y, y1)
plot(x, js)
