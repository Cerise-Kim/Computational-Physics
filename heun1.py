import numpy as np 
import matplotlib.pyplot as plt
from Heun_function import heun

def f(t, y):
    dy = -0.05 * y + 2
    return dy

t0 = 0
tf = 80
y0 = 0
h = 10
t, y = heun(f, y0, t0, tf, h)

def fa(t):
    return 40*(1-np.exp(-0.05*t))

ta = np.linspace (t0,tf)

plt.scatter (t, y, c='b',marker='o', label ='Heun Method')
plt.plot (ta,fa(ta),c='k', ls='--', label = 'True Solution')
plt.grid()
plt.title ('comparison of Tru solution and quadrature by Heun method')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend()
plt.show()