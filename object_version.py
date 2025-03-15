import numpy as np

np.random.seed(
    0
)  # for reproducibility. random.seed will generate the same random numbers every time the code is run. The 0 is an arbitrary integer used to associate the result. If we needed a second set of random numbers, we could use a different integer, i.e. 1 or 15, or whatever. Calling seed without an integer


X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]  # by convention, inputs are labeled


class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(
            n_inputs, n_neurons
        )  # random weights. The 0.1 is a scaling factor to make the weights small
        self.biases = np.zeros(
            (1, n_neurons)
        )  # biases are initialized to 0. This creates a 1x3 matrix of zeros

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # the output of the layer is the dot product of the inputs and the weights, plus the biases


# now we can easily create layers for our network

layer_1 = layer_dense(4, 5)  # 4 inputs, 5 neurons
layer_2 = layer_dense(5, 2)  # 5 inputs, 2 neurons

# in the above example, layer one take in four inputs because that is the number of features per batch in X.
# the second argument we pass is 5 but it is an arbitrary number. It is the number of neurons that we want in the layer.
# layer 2 takes in 5 inputs because that is the number of neurons in layer 1. The second argument is 2 because we want two neurons in layer 2.

# now we can pass the inputs through the layers
layer_1.forward(X)
print("layer_1 output:\n", layer_1.output)
layer_2.forward(layer_1.output)
print("layer_2 output:\n", layer_2.output)

# example layer_1 output. X has 3 feature batches and Layer 1 has 5 neurons, so the output is a 3x5 matrix. Each row is the output of a single neuron in layer 1.
# remember that with batching we get a single scalar output for each neuron in each batch regardless of the number of features in the batch.
# layer_1 output:
#  [[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
#  [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
#  [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]

# example layer_2 output. Layer two has 2 neurons, so the output is a 3x2 matrix. Each row is the output of a single neuron in layer 2.
# layer_2 output:
#  [[ 0.148296   -0.08397602]
#  [ 0.14100315 -0.01340469]
#  [ 0.20124979 -0.07290616]]
