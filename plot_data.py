import matplotlib.pyplot as plt
import numpy as np
import preprocess

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

state = preprocess.get_state(value, 0.00025)
ax.plot(time, state)

# # plot price pct chg
# ax = fig.add_subplot(4, 1, 4)
# ax.plot(time, px_pct)


# we have value array
# we need a) distribution so we can pick up proper level

# the way to choose threshold is not by looking onto real(unknown) ema distribution,
# but rather asking: how much money(kopeck) we want to earn
# plus it's a big question which wings has network prediction distribution


plt.show(True)