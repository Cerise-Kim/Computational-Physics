def euler (f, y0, t0, tf, h):
    t =[]
    t.append(t0)
    y = []
    y.append(y0)
    i = 0
    while True:
        i = i +1
        t.append(t[i-1] +h)
        if t[i] > tf:
            t[i] = tf
            h= t[i] - t[i-1]
        y.append(y[i-1] + f(t[i-1], y[i-1])*h)
        if t[i] >= tf: break
    return t, y
