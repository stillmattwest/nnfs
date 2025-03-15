import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]  # the outputs of the four neurons in the previous layer, now with batching! Each batch is a single input to the network
weights = [
    [
        0.2,
        0.8,
        -0.5,
        1.0,
    ],  # weights connecting the previous layer to the first output neuron
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]  # the weights of the three output neurons
biases = [2, 3, 0.5]  # the biases of the three output neurons

layer1_output = np.dot(inputs, np.array(weights).T) + biases

# Adding another layer.
# The output of the first layer is the input to the second layer.
# The output of the second layer is the output of the network.

weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print("layer1_output:\n", layer1_output, "\n")  # the output of the first layer

print("layer2_output:\n", layer2_output)  # the output of the output layer
# layer2_output:
#  [[ 0.5031  -1.04185  2.18525]
#  [ 0.2434  -2.7332   2.0687 ]
#  [-0.99314  1.41254  0.88425]]
