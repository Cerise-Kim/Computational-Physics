from sklearn import datasets

iris = datasets.load_iris()

import matplotlib.pyplot as plt 

fig, ax = plt.subplots()
scatter=ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend(
    scatter.legend_elements()[0],
    iris.target_names,
    loc="lower right",
    title="Classes"
)

plt.show()