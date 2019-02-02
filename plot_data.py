import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data, SECONDS_IN_DAY

vol, vol_normalized, px, px_rema, px_pct, value = load_data(20190124, 0.03)

fig = plt.figure()

time = np.linspace(1, SECONDS_IN_DAY, SECONDS_IN_DAY)

# plot volume
ax = fig.add_subplot(4, 1, 1)
ax.plot(time, vol)

# plot normalized volume
ax = fig.add_subplot(4, 1, 2)
ax.plot(time, vol_normalized)

# plot price & rema
ax = fig.add_subplot(4, 1, 3)
ax.plot(time, px)
ax.plot(time, px_rema)

# plot value
ax = fig.add_subplot(4, 1, 4)
ax.plot(time, value)

# # plot price pct chg
# ax = fig.add_subplot(4, 1, 4)
# ax.plot(time, px_pct)


# we have value array

plt.show(True)