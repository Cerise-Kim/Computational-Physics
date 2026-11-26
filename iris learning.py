import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd

from sklearn.datasets import load_iris
iris_dataset = load_iris()

type(iris_dataset)

print("key value from iris_dataset:", iris_dataset.keys())

print(iris_dataset['DESCR'][:500] + "\n...")

print("feature_names value from iris_dataset:", iris_dataset['feature_names'])
print("target_names value from iris_dataset:", iris_dataset['target_names'])

iris = pd.read_csv("iris.csv")
print(iris.head())

x_iris = iris[["sepal.length","sepal.width", "petal.length", "petal.width"]]
y_iris = iris[["variety"]]

x_iris_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_iris.columns = x_iris_columns
print(x_iris.head(5))

y_iris
y_iris['variety'].unique()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.3, random_state=0)

print("x_train 크기:{}".format(x_train.shape))
print("y_train 크기: {}".format(y_train.shape))

print("x_test 크기: {}".format(x_test.shape))
print("y_test 크리: {}".format(y_test.shape))

from sklearn.neighborsimport KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
x_train
y_train

print(y_train.shape)
print(y_train.values.ravel().shape)

knn.fit(x_train, y_train.values.ravel())

import numpy as np

Our_x = np.array([[5, 2.9, 1, 0.2]])
Our_x
prediction = knn.predict(Our_x)
print('예측: {}".format(prediction))

print("테스트 세트에 대한 예측값:\n {}".format(y_pred))

y_test_flatten =np.array(y_test).flatten()

y_test_flatten = np.array(y_test).flatten()

print("테스트 세트의 정확도: {:.2f}".format(np.mean))