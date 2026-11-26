import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

with open("C:/Users/user/Desktop/python_sj/input.csv", "r") as f:
    for line in f.readlines():
        _x, _y = [float(i) for i in line.split(" ")]
        x.append(_x)
        y.append(_y)


def g(x, y):
    new_x = []
    new_y = []
    for idx in range(len(x) - 1):
        new_x.append((x[idx] + x[idx + 1]) / 2)
        new_y.append((y[idx + 1] - y[idx]) / (x[idx +1] - x[idx]))
        return new_x, new_y
    
if __name__ == "__main__":
        fit = np.poly1d(np.polyfit(x, y, 2))
        fit_x = np.linspace(0, 1)
        fit_y=fit(fit_x)

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_title("Original function")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-2, 2)
        ax1.axes.xaxis.set_ticklabels([])
        ax1.plot(fit_x, fit_y, "k--")
        ax1.scatter(x, y)
        ax1.grid(True)

        ax2=plt.subplot(2, 1, 2)
        ax2.set_title("Differential function")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-20, 20)
        ax2.plot(fit_x[:-1], np.diff(fit_y) / (fit_x[1] - fit_x[0]), "k--")

        new_x, new_y = g(x,y)
        ax2.scatter(new_x, new_y)

        ax2.grid(True)

        plt.show()

