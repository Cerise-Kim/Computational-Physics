import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt 
x = []
y = []

with open("input.csv", "r") as f:
    for line in f.readlines():
        _x, _y = [float(i) in line.split(" ")]
        x.append(_x)
        y.append(_y)
coefficients = np.polyfit(x, y, 2)
poly_eq = np.polyld(coefficients)

y_fit = poly_eq(x)

plt.scatter(x, y, label='Original Data', color='blue')
plt.plot(x, y_fit,label-'Fitted Polynomiall', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Polynomial Fit (Degree: 2)\nEquation {coefficients[0]:.2f}x^2 + {coefficients[1]:.2f}x + {coefficients[2]:.2f}')
plt.grid(True)

plt.show


y.append
degs = [1, 2, 3, 4, 5, 8, 14, 18]
new_x =np.linspace(0, 1)
plt.figure(figsize=(8, 10))
for idx, deg in enumerate(degs):
    coef=np.polyfit(x, y, deg)
    fit=np.polyld(coef)
    sst=np.sum(np.power(y - np.average(y), 2))
    ssr = np.sum(np.power(y - fit(x), 2))
    sqr=1 - (ssr / sst)
    print(f"======= DEG: {deg}, R^2:{sqr:.3f} =======")
    print(coef)

    ax=plt.subplor(4, 2, idx + 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.text(0.05, -0.9, f"DEG: {deg}, $R^2$:{sqr:.3f}", fontsize=10)
    ax.plot(x, y, ".k")
    ax.plot(new_x, fit(new_x), "--r", label="DEG: {}".format(deg))
plt.savefig("01_data_fitting.png", bbox_inches="tight")
plt.show()