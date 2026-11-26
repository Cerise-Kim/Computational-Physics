import numpy as np
from matplotlib import pyplot as plt

print(np.pi)

N =100

X = np.linspace(0, 2*np.pi, N).reshape(N, 1)
print(X[:5])

np.random.seed(10)

idx = np.random.permutation(N)
print(idx)

shuffle_X = X[idx]
shuffle_y = y[idx]

print(shuffle_X[:3])
print(shuffle_y[:3])

