import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return 3 * x- np.cos(x)

def df(x):
    return 3 + np.sin(x)

def newton_method(x0, f, df):
    return x0 - (f(x0) / df(x0))*0.2

if __name__=="__main__":
    x = np.linspace(0, 1)
    solution_0 = 0.5
    solutions= []
    for idx in range(11):
        solutions_0 = newton_method(solution_0, f, df)
        solutions.append(solution_0)
    plt.figure()
    plt.stem(
        solutions,
        np.cos(np.array(solutions)),
        linefmt="k--",
        markerfmt="ko",
        basefmt=" ",
        label=f"[{idx: 2d}] Newton method",
    )
    plt.plot(x, 3*x, label="$3x$")
    plt.plot(x, np.cos(x), label="$\cos x$")
    plt.xlim(0, 1)
    plt.ylim(0, 2.5)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"newton_method_1.png", bbox_inches="tight")
    plt.show()