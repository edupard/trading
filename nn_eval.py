from load_data import load_data, SECONDS_IN_DAY, BEG_IDX, END_IDX
from simple_network import RnnNet, defaultRnnNetConfig
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import trading

vol, vol_normalized, px, px_rema, px_pct, value = load_data(20190124, 0.03)

F = 2

config = defaultRnnNetConfig()
config.FEATURES_DIM = F
config.DROPOUT_KEEP_RATE = 1.0

graph = tf.Graph()
sess = tf.Session(graph=graph)

net = RnnNet(config, graph)

wp = "test"
net.build_wp(wp)
net.build_sp()

net.load_wp(sess, wp, 7)

LEN = END_IDX - BEG_IDX
input = np.zeros((1, LEN, 2), dtype=np.float32)
input[0, :, 0] = vol_normalized[BEG_IDX:END_IDX]
input[0, :, 1] = px_pct[BEG_IDX:END_IDX]
labels = np.zeros((1, LEN), dtype=np.float32)
labels[0,:] = value[BEG_IDX:END_IDX]
mask = np.ones((1, LEN), np.float32)
seq_len = np.full((1,), LEN).reshape((-1,))

state = net.zero_wp_state(wp, 1)
state, sse, returns = net.eval_wp(sess, wp, state, input, labels, mask, seq_len)

prediction = np.zeros(value.shape)
prediction[BEG_IDX: END_IDX] = returns[0, :, 0]


fig = plt.figure()

time = np.linspace(1, SECONDS_IN_DAY, SECONDS_IN_DAY)

pos = trading.get_position(prediction, 0.00025)
pnl = trading.get_pnl(pos, px)

trades = trading.get_trades(pos)
volume = np.sum(np.abs(trades[np.nonzero(trades)]))

# plot volume
ax = fig.add_subplot(4, 1, 1)
ax.plot(time, prediction)
ax.plot(time, value)

# plot normalized volume
ax = fig.add_subplot(4, 1, 2)
ax.plot(time, pos)

# plot price & rema
ax = fig.add_subplot(4, 1, 3)
ax.plot(time, px)
ax.plot(time, px_rema)

# plot value
ax = fig.add_subplot(4, 1, 4)
ax.plot(time, pnl)

print( "Volume : %d" % volume)

plt.show(True)