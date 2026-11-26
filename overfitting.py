from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1, 11).reshape(10, 1)
y = np.array ([7, 8, 7, 13, 16, 15, 19, 23, 18, 21]).reshape(10, 1)

plt.plot(X, y, 'ro')

axes = plt.gca()
axes.set_ylim([0, 30])

plt.show()

reg = LinearRegression()
reg.fit(X, y)

w0 = reg.intercept_
w1 = reg.coef_

print('w0: ', w0 )
print('w1: ', w1)

y_pred = reg.predict(X)

plt.plot(X, y, 'ro')
plt.plot(X, y_pred)

axes = plt.gca()
axes.set_xlim([0, 11])
axes.set_ylim([0, 30])
plt.show()

X_2 = np.c_[X,X**2]
print(X_2)

reg = LinearRegression()
reg.fit(X_2, y)

w0 = reg.intercept_
w = reg.coef_

print('w0: ', w0)
print('w1: ', w[0][0])
print('w2: ', w[0][1])

print(reg.coef_)

y_pred_2 = reg.predict(X_2)

plt.plot(X, y, 'ro')
plt.plot(X_2[:,0], y_pred_2)

axes = plt.gca()
axes.set_xlim([0, 11])
axes.set_ylim([0, 30])
plt.show()

X_3 = np.c_[X, X**2, X**3]
print(X_3)

from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 3, include_bias = False)
X_3_poly = polynomial.fit_transform(X)

reg = LinearRegression()
reg.fit(X_3_poly, y)

w0 = reg.intercept_
w= reg.coef_
print(w)

print('w0: ', w0)
print('w1: ', w[0][0])
print('w2: ', w[0][1])
print('w3: ', w[0][2])

x_2 =np.array([2, 4, 8])
print(x_2)

y_2 = (np.array([2, 4, 8,]) @ w.reshape(-1,1)) + w0
print(y_2)

print(w[0][2] * 8 + w[0][1] *4+w[0][0]*2+7.2)

y_pred_3=reg.predict(X_3_poly)

plt.plot(X, y, 'ro')
plt.plot(X, y_pred_3)

axes = plt.gca()
axes.set_xlim([0, 11])
axes.set_ylim([0, 30])
plt.show()

X_new=np.arange(-5, 15,0.01).reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree = 3, include_bias = False)
X_new_3 = polynomial.fit_transform(X_new)

y_pred_3 = (X_new_3 @ w.reshape(-1, 1)) +w0
plt.plot(X_new, y_pred_3)

plt.plot(X, y, 'ro')
plt.plot(X_new, y_pred_3)

axes = plt.gca()
axes.set_xlim([0, 11])
axes.set_ylim([0, 30])
plt.show()

polynomial = PolynomialFeatures(degree = 7, include_bias = False)
X_7 = polynomial.fit_transform(X)
X_7

reg = LinearRegression()
reg.fit(X_7, y)

w0 = reg.intercept_
w = reg.coef_

print(w, "\n")

print('w0: ', w0 )
print('w1: ', w[0][0])
print('w2: ', w[0][1])
print('w3: ', w[0][2])
print('w4: ', w[0][3])
print('w5: ', w[0][4])
print('w6: ', w[0][5])
print('w7: ', w[0][6])

y_2 = (np.array([2,4,8,16,32,64,128]) @ w.reshape(-1,1)) +w0
y_2

print(w[0][6] * 128 + w[0][5] * 64 +w[0][4] * 32 + w[0][3] * 16 + w[0][2] * 8 + w[0][1] * 4  + w[0][0] * 2 + w0)

print([ "{:0.3f}".format(x) for x in reg.coef_.reshape(-1,)])
y_7 = (X_7 @ w.reshape(-1,1)) + w0

plt.plot(X, y, 'ro')
plt.plot(X, y_7)

axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()

X_new = np.arange(1,10.5, 0.01).reshape(-1,1)

polynomial = PolynomialFeatures(degree = 7,include_bias = False)
X_new_7 = polynomial.fit_transform(X_new)

y_7_new = (X_new_7 @ reg.coef_.reshape(-1,1)) + reg.intercept_

plt.plot(X, y, 'ro')
plt.plot(X_new, y_7_new)

axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()

from sklearn.linear_model import Ridge

ridge =Ridge(alpha=1.0)

ridge.fit(X_7, y)
y_pred_ridge = ridge.predict(X_7)

w0 = ridge.intercept_w = ridge.coef_.reshape(1,-1)

print(w,"\n")

print('w0: ', w0 )
print('w1: ', w[0][0])
print('w2: ', w[0][1])
print('w3: ', w[0][2])
print('w4: ', w[0][3])
print('w5: ', w[0][4])
print('w6: ', w[0][5])
print('w7: ', w[0][6])

print([ "{:0.3f}".format(x) for x in w.reshape(-1,)])

y_pred_ridge =ridge.predict(X_new_7)

plt.plot(X, y, 'ro')
plt.plot(X_new, y_pred_ridge)

axes = plt.gca()
axes.set_xlim([0, 11])
axes.set_ylim([0, 30])
plt.show()

plt.figure(figsize=(13, 7))

plt.plot(X, y, 'ro')
plt.plot(X_new, y_7_new)
plt.plot(X_new, y_pred_ridge)

plt.legend(['Original', 'WithoutRidge', 'Ridge Regression'])

axes =plt.gca()
axes.set_ylim([0, 30])
plt.show()