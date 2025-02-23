# the output layer is the right-most layer of the NN. In many ways, those nodes are identical to the others in previous layers. The formula for calculating their output is the same.

# in this example we have three neurons in our output layer, and four in the previous layer. That means each output neuron needs four inputs, four weights, and one bias.

inputs = [1,2,3,2.5]
inputs2 = [0.2,0.8,-0.5,1.0]
inputs3 = [-0.26,-0.27,0.17,0.87]

weights1 = [0.2,0.8,-0.5]
weights2 = [0.2,0.8,-0.5,1.0]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2= 3
bias3 = 0.5

sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0

def get_sum(inputs:list[float],weights:list[float],bias:float):
    sum = 0
    for index,input in enumerate(inputs):
        sum += input * weights[index]



output = sum + bias

print("output:",output) #2.3
