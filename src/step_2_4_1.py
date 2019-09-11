import numpy as np

w = np.array([[1], [2], [3]])
x = np.array([[3], [2], [1]])
b = 1.0
# print(w.dot(x.T) + b)
print(x.shape)
print(w.T.dot(x) + b)
# print(np.dot(w, x) + b)

x1 = np.array([1, 2, 3])
print(x1.shape)
w1 = np.array([3, 2, 1])
# print(w1.dot(x1.T) + b)
# print(w1.T.dot(x1) + b)
print(np.dot(w1, x1) + b)

x2 = np.array([[1, 2, 3]])
w2 = np.array([[3, 2, 1]])
print(x2.shape)
print(w2.dot(x2.T) + b)
# print(w2.T.dot(x2) + b)
# print(np.dot(w2, x2) + b)
