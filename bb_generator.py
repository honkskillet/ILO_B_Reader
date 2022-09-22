# %matplotlib  widget 
# %matplotlib ipympl 
#widget

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor

fig, ax = plt.subplots(figsize=(9,6))
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(3*x)
ax.plot(x, y)

cursor = Cursor(ax, horizOn=True, vertOn=True, color='Green', useblit=True)
plt.show()