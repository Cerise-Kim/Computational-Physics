import numpy as np
import matplotlib.pyplot as plt

G = 9.81
m = 1
c = 0.1

def terminal_velocity(t, v):
    dvdt = (m*G-c*(v**2))/m
    return dvdt

def euler(f, x0, xmax, y0, h):
    x = x0
    y = y0
    xsol = np.array([x])
    ysol = np.array([y])
    n = int((xmax-x0)/h)
    for i in range(n):
        y = y + h*f(x, y)
        x = x = h
        xsol =np.append(xsol, x)
        ysol =np.append(ysol, y)
    return xsol, ysol
h = [0.5, 0.1, 0.01]
x1, y1 = euler(terminal_velocity, 0, 5, 0, h[0])
x2, y2 = euler(terminal_velocity, 0, 5, 0, h[1])
x3, y3 = euler(terminal_velocity, 0, 5, 0, h[2])
plt.plot(x1, y1, color = 'red', linestyle ='dotted', label=r'h=0.5')
plt.plot(x2, y2, color = 'green', linestyle = '--', label=r'h=0.1')
plt.plot(x3, y3, color = 'blue', linestyle ='-', label=r'h=0.01')
plt.xlabel(r'$t$')
plt.ylabel(r'velocity')
plt.ylim([-0.5, 12])
plt.title("Terminal Velocity")
plt.legend()
plt.show()