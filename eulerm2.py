import numpy as np 
import matplotlib.pyplot as plt
from Euler_m_function import eulerm
g=9.801
m= 10
b = 5
k1 =1
k2=1
x1eq=0
x2eq=0
v0=0
x0=1

def f(t, y):
    v=y[0]
    x=y[1]
    dy=np.zeros(2)
    dy[0] = 1/m * (-b*v - k1*(x-x1eq)) - k2*(x-x2eq)
    dy[1] = v
    return dy

t0=0
tf=20
y0=np.array([v0, x0])
h=0.01
t, y=eulerm(f, y0, t0, tf, h)

plt.plot(t, y[:, 1], c='k', label='displacement [m]')
plt.plot(t, y[:, 0], c='k', ls='--', label='velocity [m/s]')
plt.grid()
plt.title('Eulerm to solve  the 2ndorder diff eqs')
plt.xlabel('time [s]')
plt.ylabel('displacement & velocity')
plt.legend()
plt.show()