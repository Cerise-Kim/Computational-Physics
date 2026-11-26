import numpy as np
from matplotlib import pyplot as plt

print(np.pi)

N =100

X = np.linspace(0, 2*np.pi, N).reshape(N, 1)
print(X[:5])

y = np.sin(X)
plt.plot(X, y)
plt.scatter(X, y)

np.random.seed(1)
y += 0.3*np.random.rand(N,1)

plt.scatter(X, y)
plt.title('sine wave noise data')
plt.show()

y = np.sin(X)
plt.plot(X, y)

np.random.seed(1)
y += 0.3*np.random.rand(N, 1)

plt.scatter(X, y)
plt.title('sine wave data')
plt.show()

np.random.seed(10)

idx = np.random.permutation(N)
print(idx)

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

n=int(N*(3/4))

X_train = shuffle_X[:n]
y_train = shuffle_y[:n]

X_test =shuffle_X[n:]
y_test = shuffle_y[n:]

plt.scatter(X_train, y_train, label='train')
plt.legend()
plt.title('sine wave train data')
plt.show()

plt.scatter(X_test, y_test, label='test')
plt.legend()
plt.title('sine wave test data')
plt.show()

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(X_train, y_train)

print('w0: ', reg.intercept_)
print('w1: ', reg.coef_)

reg.fit(X_test, y_test)

print('W0: ', reg.intercept_)
print('w1:', reg.coef_)

Pred = reg.predict(X_train)

plt.figure()
plt.scatter(X_train, Pred, label='pred')
plt.scatter(X_train, y_train, label='target')
plt.legend()
plt.title('train data')
plt.show()

plt.figure()
plt.scatter(X_test, reg.predict(X_test), label='pred')
plt.scatter(X_test, y_test, label='target')
plt.legend()
plt.title('test data')
plt.show()