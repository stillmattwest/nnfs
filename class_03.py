import numpy as np

inputs = [1, 2, 3, 2.5]  # the outputs of the four neurons in the previous layer
weights1 = [
    0.2,
    0.8,
    -0.5,
    1.0,
]  # the weights connecting the previous layer to the first output neuron
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weights1, weights2, weights3]  # the weights of the three output neurons
biases = [2, 3, 0.5]  # the biases of the three output neurons

outputs = [0.0, 0.0, 0.0]  # the outputs of the three neurons in the output layer


def get_output(inputs: list[float], weights: list[float], bias: float):
    sum = 0
    for index, input in enumerate(inputs):
        sum += input * weights[index]
    output = sum + bias
    return output


for output in outputs:
    for index, weight in enumerate(weights):
        outputs[index] = get_output(inputs, weight, biases[index])


print("outputs:", outputs)  # [4.8,1.21,2.385]

### A dot product for a single neuron in two ways

# manual

bias = 2

dot_prod_manual = 0

for index, input in enumerate(inputs):
    # get the sum of the products of all vector pairs
    w = weights1[index]
    dot_prod_manual += input * w

# after we have the sum of products, add the neuron's bias
dot_prod_manual += bias

print("dot prod manual:", dot_prod_manual)

dot_prod_magic = np.dot(inputs, weights1) + bias

print("dot prod magic:", dot_prod_magic)
