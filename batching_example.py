import numpy as np

inputs = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]] # the outputs of the four neurons in the previous layer, now with batching! Each batch is a single input to the network
weights = [
    [0.2, 0.8, -0.5, 1.0],  # weights connecting the previous layer to the first output neuron
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] # the weights of the three output neurons
biases = [2, 3, 0.5] # the biases of the three output neurons

output = np.dot(inputs,np.array(weights).T) + biases
print("output batch:\n",output) # the output of the output layer

# output batch:
# [[ 4.8    1.21   2.385]
#  [ 8.9   -1.81   0.2  ]
#  [ 1.41   1.051  0.026]]


#np.array is a function that takes in a LOL and returns a numpy array. .T is the transpose method of a numpy array.

