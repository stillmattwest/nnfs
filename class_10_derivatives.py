import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x * 2


x = np.array(range(5))
y = f(x)

print("x:", x)
print("y:", y)

plt.plot(x, y)
plt.show()
