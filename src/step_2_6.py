import numpy as np

x = np.eye(4, 3)
w = np.array([[1], [2], [3]])

print(x)
print(x.shape)
print(w)
print(w.shape)

print(x.dot(w))

x = np.eye(3, 4)
w = np.array([[1], [2], [3]])

print('##### 2 #####')
print(x)
print(x.shape)
print(w)
print(w.shape)
print(w.T.dot(x).T)

x = np.eye(3, 4)
w = np.array([[1], [2], [3]])
w = w.reshape(1, 3)

print('##### 3 #####')
print(x)
print(x.shape)
print(w)
print(w.shape)
print(w.dot(x).T)

x = np.eye(4, 3)
w = np.array([[1], [2], [3]])
w = w.reshape(1, 3)

print('##### 4 #####')
print(x)
print(x.shape)
print(w)
print(w.shape)
print(x.dot(w.T))
