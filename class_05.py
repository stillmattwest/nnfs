import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# X = [[1,2,3,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]] # the outputs of the four neurons

X, y = spiral_data(100, 3)

# in the line above, spiral_data is a function that generates a dataset of points that are arranged in a spiral pattern.
# The first parameter is the number of points per class, and the second parameter is the number of classes.
# The function returns two arrays: X, which contains the coordinates of the points, and y, which contains the class labels of the points.

# vocabulary: samples are individual datapoints and a class is a group of samples.
# In this case, each sample is a point in 2D space, and each class is a spiral in that space.


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


layer1 = Layer_Dense(
    2, 5
)  # spirals (like anything in 2D space) have 2 coordinates, and we want 5 (an arbitrary number) neurons in the first layer
activation1 = Activation_ReLU()
layer1.forward(X)

print("layer1 data pre-activation:", layer1.output)

activation1.forward(layer1.output)
print("layer1 data post-activation:", activation1.output)
