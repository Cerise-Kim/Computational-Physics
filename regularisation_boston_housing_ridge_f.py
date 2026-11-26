import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge

boston = fetch_openml(name="boston", version=1, as_frame=True)
X_df = boston.data
y_data = boston.target.to_numpy(dtype=float)

print(X_df)
print("원 특성 수:", X_df.shape[1], "행:", X_df.shape[0])

poly =PolynomialFeatures(degree=2, include_bias=False)
X_data = poly.fit_transform(X_df.values)
feature_names = poly.get_feature_names_out(X_df.columns)

print("X data 의 shape:", X_data.shape)
print("y data의 shape:", y_data.shape)
print(pd.DataFrame(X_data, columns=feature_names).head())

N = X_data.shape[0]
np.random.seed(3)
idx = np.random.permutation(N)
print("셔플 인덱스 길이:", len(idx))

shuffle_X = X_data[idx]
shuffle_y =y_data[idx]
n= int(N*(3/4))
X_train=shuffle_X[:n]
y_train = shuffle_y[:n]
X_test = shuffle_X[n:]
y_test = shuffle_y[n:]

print("X_train:", X_train.shape, "X_test:", X_test.shape, "합:", X_train.shape[0] + X_test.shape[0])

X_mean =X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1.0

X_train_s = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

def MSE(real, predict):
    return ((real -predict) ** 2).mean()

def get_top_bottom(data, count):
    idx_sorted = np.argsort(data)
    top_idx = idx_sorted[-count:]
    bottom_idx = idx_sorted[:count]
    top_vals = [data[i] for i in top_idx]
    bottom_vals = [data[i] for i in bottom_idx]
    return top_vals, bottom_vals

print("\n=== OLS(LinearRegression) ===")
print("Training R^2 score:{:0.2f}".format(reg.score(X_train, y_train)))
print("Test R^2 score:{:0.2f}".format(reg.score(X_test, y_test)))
print('Mean squared error:{:0.2f}'.format(MSE(y_test, y_pred)))

W= reg.coef_
top_3_idx=np.argsort(W)[-3:]
bottom_3_idx = np.argsort(W)[:3]
print("OLS 가장 높은 W 3개 =>", ["{:0.2f}".format(W[i]) for i in top_3_idx])
print("OLS 가장 낮은 W 3개 =>", ["{:0.2f}".format(W[i]) for i in bottom_3_idx])