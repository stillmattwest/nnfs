# the output layer is the right-most layer of the NN. In many ways, those nodes are identical to the others in previous layers. The formula for calculating their output is the same.

# in this example we have three neurons in our output layer, and four in the previous layer. That means each output neuron needs four inputs, four weights, and one bias.

inputs = [1,2,3,2.5] # the outputs of the four neurons in the previous layer
weights1 = [0.2,0.8,-0.5,1.0] # the weights connecting the previous layer to the first output neuron
weights2=[0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]
weights = [weights1,weights2,weights3] # the weights of the three output neurons
biases = [2,3,0.5] # the biases of the three output neurons

outputs = [0.0,0.0,0.0] # the outputs of the three neurons in the output layer

def get_output(inputs:list[float],weights:list[float],bias:float):
    sum = 0
    for index,input in enumerate(inputs):
        sum += input * weights[index]
    output = sum + bias
    return output

for output in outputs:
    for index,weight in enumerate(weights):
        outputs[index] = get_output(inputs,weight,biases[index])
    

print("outputs:",outputs) #[4.8,1.21,2.385]
