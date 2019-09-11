import numpy as np
from numpy import linalg

x = np.array([[1, 60], [1, 50], [1, 75]])
y = np.array([[10], [7], [12]])

step1 = x.T.dot(x)
step2 = linalg.inv(step1)
step3 = step2.dot(x.T)
step4 = step3.dot(y)

print(step1)
print(step2)
print(step3)
print(step4)
print(2225 / 950, 185 / 950)
print(y)
