import matplotlib.pyplot as plt
import numpy as np
import create_data as cdata

X,y = cdata.create_data(100,3)

plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()

