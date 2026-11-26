import numpy as np 
import matplotlib.pyplot as plt
from Euler_m_function import eulerm
def f(t, y):
    dy = np.zeros(2)
    dy[0] = -2*y[0]**2 + 2*y[0] + y[1] -1
    dy[1] = -y[0] -3*y[1]**2+2*y[1] +2
    return dy

t0 = 0
tf = 2
y0 = np.array([2., 0.])
h = 0.01
t, y =eulerm(f, y0 , t0 ,tf, h)

plt.plot(t, y[:, 0], c='k', label='x1')
plt.plot(t, y[:, 1], c='k', ls='--', label='x2')
plt.grid()
plt.title('Modified Euler method to solve multi diff eqs')
plt.xlabel('$t$')
plt.ylabel('x1 and x2')
plt.legend()
plt.show()