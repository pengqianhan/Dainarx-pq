import numpy as np

x =    np.array([ 1.18889388 ,-3.37011189,3.18128494])
y =    np.array([ 0.61141107, -2.18160506, 2.56946204])

z = np.dot(x, y) / (np.linalg.norm(x * np.linalg.norm(y)))

print(z)

