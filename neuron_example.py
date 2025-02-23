# every neuron has inputs, which come from the previous layer
# every neuron has a unique connection to each neuron in the previous layer
# The neurons in one layer have outputs that become the inputs of other neurons
inputs = [1.2,5.1,2.1]
# every input has an unique weight associated with it.
# we need three weights since we have three inputs
weights = [3.1,2.1,8.7]
# every neuron also has a unique bias
bias = 3

# we calculate the sum of all inputs multiplied by their associated weight, then we add (not multiply) the bias to the sum.
sum = 0

for index,input in enumerate(inputs):
    sum += input * weights[index]

output = sum + bias

print("output:",output)