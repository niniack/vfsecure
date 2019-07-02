import numpy as np
a = [[[1,2,3], [4,5,6], [1,2,3]],[[1,2,3], [4,5,6], [1,2,3]]]
b = [1,2,3]

print(a)
print("\n")
a = np.insert(a, 3, -1, axis=2)
print(a)

a[0,:,3] = 5
print("\n")
print(a)
