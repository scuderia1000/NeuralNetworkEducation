import numpy as np

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
print(w)
