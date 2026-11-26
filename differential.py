import matplotlib.pyplot as plt 
import numpy as np

def f(x):
    return 3*x**2+2*x+6

def g(func, x):
    y = []
    h = 0.01
    for _x in x:
        y.append((func(_x+h) - func(_x)) / h)
    return np.array(y)
    
if __name__== "__main__":
    x=np.linspace(0, 10)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("Original fuction")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 400)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.plot(x, f(x))
    ax1.grid(True)
    ax2=plt.subplot(2, 1, 2)
    ax2.set_title("Differential function")
    ax2.set_xlim(0,10)
    ax2.set_ylim(0,200)
    ax2.plot(x, g(f, x))
    ax2.grid(True)
    plt.show()