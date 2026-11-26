from sklearn import datasets
iris = datasets.load_iris()

print("Iris DatasetOverview\n")
print("Keys:", iris.keys(), "\n")

print("Description:\n", iris['DESCR'][:2000], "\n...")
print("\nFeature Names:\n", iris['feature_names'], "\n")
print("Target Names:\n", iris['target_names'], "\n")

print("Data (features):\n", iris['data'], "\n")
print("Target (class labels):\n", iris['target'])

import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris_df=pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
iris_df['targe'] = iris['target']

iris_df.to_csv("iris_dataset.csv", index=False)

