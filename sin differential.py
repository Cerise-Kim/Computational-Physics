import matplotlib.pyplot as plt 
import numpy as np 
def f(x):
    return np.sin(x)

def g(func,x):
    y = []
    h = 0.01
    for _x in x:
        y.append((func(_x +h) - func(_x)) / h)
        return np.array(y)
    

    x=np.linspace(0, np.pi * 10, 500)
    plt.plot(x, f(x), "k--", label="sin(x)")
    y=g(f,x)
    plt.plot(x, y, "r", label="Derivative of sin(x)")
    
    plt.grid(True)
    plt.legend()
    plt.show()