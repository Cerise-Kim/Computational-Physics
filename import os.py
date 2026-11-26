import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
iris_df['target']=iris['target']

iris_df.to_csv("iris_dataset.csv", index=False)