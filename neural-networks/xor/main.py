import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

W1 = np.random.randn(2,2)
W2 = np.random.randn(2,1)

def sigmoid(x): return 1/(1+np.exp(-x))

for _ in range(5000):
    # forward
    h = sigmoid(X @ W1)
    y_hat = sigmoid(h @ W2)

    # backward
    error = y_hat - y
    dW2 = h.T @ error
    dW1 = X.T @ ((error @ W2.T) * h * (1-h))

    W1 -= 0.1 * dW1
    W2 -= 0.1 * dW2

print(y_hat)