import numpy as np 

def eulerm(f, y0, t0, tf, h):
    n = len(y0)
    m = int((tf -t0) / h) + 1
    t = np.zeros(m)
    y = np.zeros((m, n))
    y[0,:] = y0
    for i in range ( 1, m):
        t[i] = t0 + i*h
        if t[i] > tf:
            t[i]= tfh = t[i] -t[i-1]
        y[i] = y[i-1,:] + f(t[i-1], y[i-1,:])*h
    return t, y