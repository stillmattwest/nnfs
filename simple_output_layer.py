# the output layer is the right-most layer of the NN. In many ways, those nodes are identical to the others in previous layers. The formula for calculating their output is the same.

# in this example we have three neurons in our output layer, and four in the previous layer. That means each output neuron needs four inputs, four weights, and one bias.

inputs = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]] # the outputs of the four neurons in the previous layer, now with batching! Each batch is a single input to the network
weights = [
    [0.2, 0.8, -0.5, 1.0],  # weights connecting the previous layer to the first output neuron
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] # the weights of the three output neurons
biases = [2, 3, 0.5] # the biases of the three output neurons

output = np.dot(inputs, np.array(weights).T) + biases
print("output:", output) # [[ 4.8   1.21  2.385]]



print("outputs:", outputs) # [4.8, 1.21, 2.385]
