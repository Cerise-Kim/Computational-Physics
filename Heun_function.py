import numpy as np 

def heun (f, y0, t0, tf, h, ea=1.e-7):
    n = int ((tf -t0) / h)
    t=np.zeros (n)
    y=np.zeros (n)
    for i in range (n):
        t[i] = i*h
    if t[n-1] < tf:
        t = np.append (t, tf)
        y = np.append (y, 0)
    for i in range (n):
        h = t[i+1] - t[i]
        y1 = y[i] + f(t[i], y[i] * h)
        while True:
            fbar = (f(t[i], y[i]) + f(t[i+1], y1))/2
            y2 = y[i] + fbar * h
            if abs ((y1 - y2)/y2) < ea: break
            y1 = y2
        y[i+1] = y2
    return t, y