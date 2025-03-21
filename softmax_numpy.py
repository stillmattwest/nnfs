import math
import numpy as np
import nnfs

nnfs.init() # this function initializes the random number generator in the numpy library and makes our output reproducible.

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print("exponentiated values:", exp_values) # numpy automatically uses Euler's number as the base for exponentiation. If you want to use a different base, you can use the math library (math.power(base, exponent)).
print("sum of exponentiated values:", sum(exp_values))


norm_values = exp_values / np.sum(exp_values)

print("normalized exponentiated values:", norm_values)
print("sum of normalized values:", sum(norm_values))

