import numpy as np
import nnfs

nnfs.init()

X = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]] # the outputs of the four neurons in the previous layer, now with batching! Each batch is a single input to the network

inputs = [0,2,-1,3,3,-2.7,2.2,-100]
output = []

# The Rectified Linear Unit (ReLU) activation function is this simple. It returns the input if the input is greater than zero, and zero otherwise.
# ReLU is the most popular activation function in deep learning, and is used in most convolutional neural networks (CNNs).
# ReLU is used because it is computationally efficient, and it allows the network to learn faster and perform better.
###
# for i in inputs:
#     output.append(max(0,i)) # ReLU activation function. Anything less than zero is set to zero.

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
#print(layer2.output)