import numpy as np

np.random.seed(1)
y += 0.3*np.random.rand(N,1)

plt.scatter(X, y)
plt.title('sine wave noise data')
plt.show()