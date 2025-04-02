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
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    # inherits from the Loss class
    def forward(self, y_pred, y_true):
        # number of samples in the batch
        samples = len(y_pred)

        # clip data to prevent div by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # our labels will either be a vector if we're using sparse labels or a matrix if we're using one-hot encoded labels (a list of one hot vectors)
        # probabilities for taget values if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # probabilities for target values if one-hot encoded labels. We'll need to mask these by summing the output. All the "cold" values will be 0.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# our dataset
X, y = spiral_data(samples=100, classes=3)

# passing the dataset through the network

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])  # print the first 5 samples


"""
[[0.33333334 0.33333334 0.33333334]
 [0.33331734 0.3333183  0.33336434]
 [0.3332888  0.33329153 0.33341965]
 [0.33325943 0.33326396 0.33347666]
 [0.33323312 0.33323926 0.33352762]]
"""
# This is the output of the network. Each row is a sample, and each column is a class.
# We're looking at a probability distribution, so the values in each row should sum to 1.
# The values are close to 0.33 because we have 3 classes, so the network is guessing randomly.

# What we need to do from here is to add a loss function to the network, so we can train it to make better guesses.

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print(
    "Loss:", loss
)  # approx 1.098 which is the expected value for a random guess with three classes (-log(1/3) = 1.098)
