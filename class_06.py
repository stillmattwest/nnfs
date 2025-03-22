import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLu is pretty simple. It just sets all negative values to zero and keeps all positive values the same. See notes.

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax is a bit more complicated. It takes the exponential of each value in the input array, then divides each value by the sum of all the exponentials.
# This gives us a probability distribution, which is useful for classification problems.
# Softmax is used as the final activation function in neural networks that are used for classification problems.

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# our dataset
X, y = spiral_data(samples = 100, classes = 3)

# passing the dataset through the network

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5]) # print the first 5 samples


'''
[[0.33333334 0.33333334 0.33333334]
 [0.33331734 0.3333183  0.33336434]
 [0.3332888  0.33329153 0.33341965]
 [0.33325943 0.33326396 0.33347666]
 [0.33323312 0.33323926 0.33352762]]
'''
# This is the output of the network. Each row is a sample, and each column is a class.
# We're looking at a probability distribution, so the values in each row should sum to 1.
# The values are close to 0.33 because we have 3 classes, so the network is guessing randomly.

# What we need to do from here is to add a loss function to the network, so we can train it to make better guesses.
# We'll do that in the next lesson.