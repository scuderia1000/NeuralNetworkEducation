import urllib
from urllib import request
import numpy as np
from numpy import linalg

# Sample Input:
#
# https://stepic.org/media/attachments/lesson/16462/boston_houses.csv
# tmpFileName = 'https://stepic.org/media/attachments/lesson/16462/boston_houses.csv'
# fname = tmpFileName  # read file name from stdin
fname = input()  # read file name from stdin
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

y = data[:, 0].reshape((-1, 1))

xFakeColumn = np.ones_like(y)
x = np.hstack((xFakeColumn,  data[:, 1:]))

step1 = x.T @ x
step2 = linalg.inv(step1) @ x.T
step3 = step2 @ y
result = np.around(step3.ravel(), decimals=4)

print(' '.join(str(x) for x in result))

