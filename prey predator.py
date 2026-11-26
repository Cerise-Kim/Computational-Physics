import numpy as np 
import matplotlib.pyplot as plt

a =3
b  =2
c=1
d=2

def lveq(t, y):
    dydt0=a*y[0] - b*y[0]*y[1]
    dydt1=-c*y[1]+d*y[0]*y[1]
    return np.array([dydt0, dydt1])

def RK4(f, h, t0, tmax,y0):
    n = int((tmax - t0)/h)
    t=t0
    y=y0
    m = len(y0)
    tsol=np.zeros(n+1)
    ysol = np.zeros([n+1, m])
    tsol[0] = t
    ysol[0, :] = y
    for i in range(n):
        k1 = f(t, y)
        k2=f(t+0.5*h, y+0.5*h*k1)
        k3=f(t+0.5*h, y+0.5*h*k2)
        k4=f(t+h, y+h*k3)
        y = y+ h*(k1 + 2*k2 + 2*k3 +k4)/6
        t=t+h
        tsol[i+1]=t
        ysol[i+1, :]=y
    return tsol, ysol

h=0.01
t0, tmax = 0, 20
y0 = np.array([1.9, 1.7])
t, y = RK4(lveq,h, t0, tmax, y0)
plt.plot(t, y[:, 0], label = r'$x$'' (prey)')
plt.plot(t, y[:, 1], label = r'$y$'' (predator)')
plt.xlabel(r'$t$')
plt.ylabel(r'$x, y$''(poplulation)')
plt.legend()
plt.title("Lotka-Volterra Equation")
plt.show()
