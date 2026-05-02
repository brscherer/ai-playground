import numpy as np

# [temp, rain, energy, time]
X = np.array([
    [25, 0, 8, 30],
    [18, 1, 5, 20],
    [22, 0, 7, 40],
])

y = np.array([[1], [0], [1]])

weights = np.random.randn(4, 1)
bias = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for _ in range(1000):
    z = X @ weights + bias
    y_hat = sigmoid(z)

    error = y_hat - y
    weights -= 0.01 * (X.T @ error)

print(y_hat)