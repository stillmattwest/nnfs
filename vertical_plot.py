import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data, spiral_data

nnfs.init()

# X, y = vertical_data(samples=200, classes=3)

X, y = spiral_data(samples=200, classes=3)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=40, edgecolors="k")
plt.title("Vertical Dataset")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
