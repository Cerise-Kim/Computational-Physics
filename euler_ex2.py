import numpy as np 
import matplotlib.pyplot as plt
from Euler_function import euler

def f(t, y):
    dy = -0.05* y +2.
    return dy

t0=0
tf = 80
y0 = 0
h = 5
t, y =euler(f, y0, t0, tf, h)

def fa(t):
    return 40*(1 - np.exp(-0.05*t))

ta = np.linspace (t0, tf)

plt.scatter (t, y, c='k', marker='o', label = 'Euler Method')
plt.plot (ta, fa(ta),c='k', ls='--', label = 'True Solution')
plt.grid()
plt.title('Comparison of True solution and quadrature by Euler method')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend()
plt.show()