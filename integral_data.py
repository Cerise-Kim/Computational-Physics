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
    popt1, pcov1 = curve_fit(particle, bins[:18], count[:18], p0=[1000, 1.0, 0.2])
    popt2, pcov2 = curve_fit(particle, bins[18:], count[18:], p0=[500, 2.0, 0.5])

    h = 0.01
    x = np.arange(0, 4, h)

    y1 = particle(x, popt1[0], popt1[1], popt1[2])
    y2 = particle(x, popt2[0], popt2[1], popt2[2])

    int_A = trapezoid(y1, x, h)
    int_B = trapezoid(y2, x, h)
    print("생성비 (A / B):", int_A / int_B)

    plt.bar(bins, count, width=0.08, color="gray", alpha=0.5)
    plt.plot(x, y1, "r--", label="Particle A")
    plt.plot(x, y2, "b--", label="Particle B")

    plt.text(2.2, 800, f"Ratio (A/B): {int_A / int_B:.2f}")
    plt.xlim(0, 4)
    plt.ylim(0, 1200)
    plt.grid(True)
    plt.legend()
    plt.show()