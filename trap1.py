import matplotlib.pyplot as plt 
import numpy as np

from Trap1_function import trap1

def f(t):
    return t * np.cos(t)

t, y = trap1(f, 0., 2)

plt.plot(t, y, label='Integral of f(t)')
plt.plot(t, f(t), label='f(t)', linestyle='--')
plt.xlabel('t')
plt.ylabel('Value')
plt.title('Plot of f(t) and its integral over time')
plt.legend()
plt.grid()
plt.show()