import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

N =100

X = np.linspace(0, 2*np.pi, N).reshape(N, 1)

y=np.sin(X)
y += 0.3*np.random.rand(N, 1)

polynomial = PolynomialFeatures(degree = 3, include_bias =False)
sin_new_feature = polynomial.fit_transform(X)

print("기존의 x[0]:",X[1])
print("Feature가 추가된 x[0]:", sin_new_feature[1])

print(sin_new_feature)

idx = np.random.permutation(N)

shuffle_X = sin_new_feature[idx]
shuffle_y = y[idx]

print(shuffle_X[:3])
print(shuffle_y[:3])

n=int(N*(3/4))

X_train = shuffle_X[:n]
y_train = shuffle_y[:n]

X_test =shuffle_X[n:]
y_test = shuffle_y[n:]

reg = LinearRegression()
reg.fit(X_train, y_train)

print('w0: ', reg.intercept_)
print('w1: ', reg.coef_)

Pred = reg.predict(X_train)

plt.figure()
plt.scatter(X_train[:,0], Pred, label='pred')
plt.scatter(X_train[:,0],y_train, label='target')
plt.legend()
plt.title('train data')
plt.show()

test_Pred = reg.predict(X_test)

plt.figure()
plt.scatter(X_test[:,0], test_Pred, label='pred')
plt.scatter(X_test[:,0], y_test, label='target')
plt.legend()
plt.title('test data')
plt.show()

X_all = shuffle_X[:]
y_all = shufle_y[:]

all_Pred = reg.predict(X_all)

plt.figure()
plt.scatter(X_all[:,0], all_Pred, label='pred')
plt.scatter(X_all[:,0], y_all, label='target')
plt.legend()
plt.title('all data')
plt.show()