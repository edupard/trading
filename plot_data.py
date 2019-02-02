import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data, SECONDS_IN_DAY

data = load_data(20190124)

fig = plt.figure()

ax = fig.add_subplot(4, 1, 1)


time = np.linspace(1, SECONDS_IN_DAY, SECONDS_IN_DAY)

ax.plot(time, data[:, 0])
ax.plot(time, data[:, 1])

ax = fig.add_subplot(4, 1, 2)
ax.plot(time, data[:, 2])


ax = fig.add_subplot(4, 1, 3)
ax.plot(time, data[:, 3])

plt.show(True)