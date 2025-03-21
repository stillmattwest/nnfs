import math

# the softmax function is used in the output layer of a neural network to make the output a probability distribution.
# the softmax function takes the output of the network and normalizes it to a probability distribution over the classes.
layer_outputs = [4.8, 1.21, 2.385]
# E = 2.71828182846 (Euler's Number)
E = math.e

exp_values = [E ** output for output in layer_outputs]
print("exponentiated values:", exp_values)

# exponentiation is a way to make all the values positive, but it also makes the values larger.
# to make the values smaller, we need to normalize them.

norm_base = sum(exp_values)
norm_values = [value / norm_base for value in exp_values]
print("normalized exponentiated values:", norm_values)
print("sum of normalized values:", sum(norm_values))

# the sum of the normalized values is 1.0, which is a property of the softmax function.
# the softmax function is used to normalize the output of a network to a probability distribution over predicted output classes.
# the output of the softmax function is a probability distribution over the classes.
