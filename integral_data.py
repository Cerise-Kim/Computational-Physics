import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


bins, count = [], []
with open("hist2.csv", "r") as f:
    for line in f.readlines():
        _b, _c = [float(i) for i in line.split(",")]
        bins.append(_b)
        count.append(_c)


def particle(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


if __name__ == "__main__":

    popt1, pcov1 = curve_fit(particle, bins[:19], count[:19])
    popt2, pcov2 = curve_fit(particle, bins[19:], count[19:], p0=[500, 2.0, 0.5])

    h = 0.01
    x = np.arange(0, 4, h)

    y1 = particle(x, popt1[0], popt1[1], popt1[2])
    y2 = particle(x, popt2[0], popt2[1], popt2[2])


    s1 = trapezoid(y1, x, h)
    s2 = trapezoid(y2, x, h)
    print("A:", s1)
    print("B:", s2)
    print("A / B:", s1 / s2)

    plt.bar(bins, count, width=0.08, color="gray", alpha=0.5)
    plt.plot(x, y1, "r--")
    plt.plot(x, y2, "b--")

    plt.text(2.5, 800, f"A / B = {s1 / s2:.3f}")
    plt.grid(True)
    plt.xlim(0, 4)
    plt.ylim(0, 1150)
    plt.show()