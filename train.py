from simple_network import RnnNet, defaultRnnNetConfig
import numpy as np

BS = 3
TS = 100
F = 2
EPOCHS = 1000

config = defaultRnnNetConfig()
config.FEATURES_DIM = F
net = RnnNet(config)

net.init_weights()

for epoch in range(EPOCHS):
    input = np.random.randint(0, 100, (BS, TS, F))
    labels = np.mean(input, 2)
    mask = np.ones((BS, TS), np.float32)
    seq_len = np.full((BS,), TS).reshape((-1,))

    state = net.zero_state(BS)
    state, sse, returns = net.fit(state, input, labels, mask, seq_len)
    print(sse)


input = np.random.randint(0, 100, (BS, TS, F))
labels = np.mean(input, 2)
mask = np.ones((BS, TS), np.float32)
seq_len = np.full((BS,), TS).reshape((-1,))

state = net.zero_state(BS)
new_state, sse, returns = net.eval(state, input, labels, mask, seq_len)

i = 0