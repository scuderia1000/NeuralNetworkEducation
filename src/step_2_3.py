import numpy as np
from src.gradient.util import sigmoid, sigmoid_prime

x = np.array([[1, 1, 1],
              [1, 1, 0],
              [1, 0, 1],
              [1, 0, 0]])
w = np.zeros((3, 1))
y = np.array([1, 0, 0, 0])

# x = np.array([[1, 1, 0.3],
#               [1, 0.4, 0.5],
#               [1, 0.7, 0.8]])
# w = np.zeros((3, 1))
# y = np.array([1, 1, 0])

for k in range(len(x)):
    delta_y = np.copy(w[0])

    for i in range(1, len(x[k, 1:]) + 1):
        delta_y += x[k, i] * w[i]

    if delta_y > 0:
        delta_y = 1
    else:
        delta_y = 0
    for i in range(len(w)):
        w[i] = w[i] + (y[k] - delta_y) * x[k, i]
# print(w)

w = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# 2 примера
deltas = np.array([[1, 2, 3],
                   [4, 5, 6]])
# print(w.T * deltas)
# print(sigmoid(1.994))
z_2 = np.array([[6],
                [6]])
X = np.array([[1],
              [0],
              [1],
              [2]])
delta_2_first = np.array([[-0.013],
                          [-0.013]])
sigm_prime_z_2 = sigmoid_prime(z_2)
deltas_2 = delta_2_first * sigm_prime_z_2
print(deltas_2)
grad_2 = X.dot(deltas_2.T)
print(grad_2)

