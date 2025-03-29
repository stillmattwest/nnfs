# categorical cross entropy

import math

softmax_output = [0.7, 0.2, 0.1] # random example of softmax output
target_output = [1, 0, 0] # random example of target output. One hot encoding

loss = (math.log(softmax_output[0]) * target_output[0] +
        math.log(softmax_output[1]) * target_output[1] +
        math.log(softmax_output[2]) * target_output[2])
print('Loss:', loss)
# Loss: -0.35667494394077323
# This is the loss for this example. The loss is negative because we're taking the log of a number less than 1.

# With one-hot encoding, we can simplify this to just the first value. This is because the other values are multiplied by 0. basically, Math.log(softmax_output[0]) * 1 + 0 + 0 = Math.log(softmax_output[0])

print('Proving this works with one-hot encoding',math.log(softmax_output[0]) * target_output[0])
# Loss: -0.35667494394077323
# This is the same loss as before. This is because the other values are multiplied by 0.

# But we actually want to use the negative of this value. This is because we want to minimize the loss, and the loss is negative. So we need to multiply by -1 to get a positive value.

# We generally want to get a value between 0 and 1, so we can use this value to calculate the accuracy of the network. This is done by taking the negative of the loss. This is because our one-hot encoding is 1 for the correct class and 0 for the other classes. So we can just take the negative of the loss to get a positive value.

print('Loss is negative log of softmax output', math.log(softmax_output[0]) * target_output[0] * -1)
# Loss: 0.35667494394077323