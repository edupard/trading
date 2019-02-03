from load_data import load_data, SECONDS_IN_DAY, BEG_IDX, END_IDX
from simple_network import RnnNet, defaultRnnNetConfig
import numpy as np
import tensorflow as tf

vol, vol_normalized, px, px_rema, px_pct, value = load_data(20190124, 0.03)

TS = 100
F = 2
EPOCHS = 100

config = defaultRnnNetConfig()
config.FEATURES_DIM = F
config.DROPOUT_KEEP_RATE = 1.0

graph = tf.Graph()
sess = tf.Session(graph=graph)

net = RnnNet(config, graph)

wp = "test"
net.build_wp(wp)
net.build_sp()

net.init_weights(sess)

LEN = END_IDX - BEG_IDX
input = np.zeros((1, LEN, 2), dtype=np.float32)
input[0, :, 0] = vol_normalized[BEG_IDX:END_IDX]
input[0, :, 1] = px_pct[BEG_IDX:END_IDX]
labels = np.zeros((1, LEN), dtype=np.float32)
labels[0,:] = value[BEG_IDX:END_IDX]
mask = np.ones((1, LEN), np.float32)
seq_len = np.full((1,), TS).reshape((-1,))


# input = np.random.randint(0, 100, (BS, TS, F))
# labels = np.mean(input, 2)
# mask = np.ones((BS, TS), np.float32)
# seq_len = np.full((BS,), TS).reshape((-1,))
# return input, labels, mask, seq_len

periods = LEN // TS

for epoch in range(EPOCHS):
    state = net.zero_wp_state(wp, 1)
    for i in range(periods):
        bi = i*TS
        ei = (i+1)*TS

        state, sse, returns = net.fit_wp(sess, wp, state, input[:,bi:ei,:], labels[:,bi:ei], mask[:,bi:ei], seq_len)
        print("%s %d %.8f" % (wp, epoch, sse))

TODO:
1. how many data we can fit
2. mask, batch size
3. what error level is acceptable -  we have original distribution
