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
