import numpy as np
import nnfs
from nnfs.datasets import vertical_data
from nn_template_01 import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax,
    Loss_CategoricalCrossentropy,
)

nnfs.init()

# create data
X, y = vertical_data(samples=100, classes=3)

# print(X[:10])
# print(y[:10])

# create model
dense1 = Layer_Dense(2, 6)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()  # activation function
dense2 = Layer_Dense(6, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()  # activation function

# create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999  # arbitrary large number
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Update weights with some small random values
    dense1.weights += np.random.uniform(
        -0.05, 0.05, size=dense1.weights.shape
    ) * np.random.randn(2, 6)
    dense1.biases += np.random.uniform(
        -0.05, 0.05, size=dense1.biases.shape
    ) * np.random.randn(6)
    dense2.weights += np.random.uniform(
        -0.05, 0.05, size=dense2.weights.shape
    ) * np.random.randn(6, 3)
    dense2.biases += np.random.uniform(
        -0.05, 0.05, size=dense2.biases.shape
    ) * np.random.randn(3)

    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # if loss is smaller than the lowest loss, save the weights and biases
    if loss < lowest_loss:
        print(
            f"New set of weights found, iteration {iteration}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
        )
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
