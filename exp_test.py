import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 1, 0.01)

fact = 6

y = np.exp(fact * -1 * x)


plt.plot(x,y )
plt.show()
