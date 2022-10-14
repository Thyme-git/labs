import numpy as np
from nn.utils import *


x = np.random.random_integers(0, 10, (2, 2, 2, 3))

print(x)

x = x.reshape((2, -1))

print(x.reshape(2, 2, 2, 3))