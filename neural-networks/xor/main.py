import numpy as np

# [temp, rain, energy, time]
X = np.array([
    [25, 0, 8, 30],
    [18, 1, 5, 20],
    [22, 0, 7, 40],
], dtype=float)

y = np.array([[1], [0], [1]], dtype=float)

# normalizing inputs
X = X / np.max(X, axis=0)

np.random.seed(42)  # reproducibility
weights = np.random.randn(4, 1)
bias = 0.0

def sigmoid(x):
    x = np.clip(x, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

learning_rate = 0.1
epochs = 2000

for epoch in range(epochs):
    z = X @ weights + bias
    y_hat = sigmoid(z)
    loss = np.mean((y - y_hat) ** 2)
    error = y_hat - y
    gradient = error * sigmoid_derivative(y_hat)

    dW = X.T @ gradient
    db = np.sum(gradient)

    weights -= learning_rate * dW
    bias -= learning_rate * db

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal predictions:")
print(y_hat)