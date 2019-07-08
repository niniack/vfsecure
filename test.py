import numpy as np
a = np.array([[[1,2,3], [4,5,6], [9,2,5]],[[1,2,3], [6,4,3], [2,8,4]]])
b = np.array([1,2,3])

a = np.reshape(a, (int(a.size/3), 3))
row, col = np.all(a == b, axis=2)
print(a[:, :])
print (row, col)
