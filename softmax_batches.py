import math
import numpy as np
import nnfs

nnfs.init() # this function initializes the random number generator in the numpy library and makes our output reproducible.

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs) # numpy handles batched data automatically

# print(np.sum(layer_outputs, axis=1, keepdims=True)) # sum of each row

# axis 0 is the vertical axis, axis 1 is the horizontal axis

# so if we gave an axis=0, we would get the sum of each column i.e 4.8+8.9+1.41, etc)

# keepdims=True keeps the dimensions of the array the same, so we can divide the array by the sum of the array

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

