from simple_network import RnnNet, defaultRnnNetConfig
import numpy as np
import tensorflow as tf

BS = 3
TS = 100
F = 2
EPOCHS = 100

config = defaultRnnNetConfig()
config.FEATURES_DIM = F
config.DROPOUT_KEEP_RATE = 1.0

graph = tf.Graph()
sess = tf.Session(graph=graph)

net = RnnNet(config, graph)

wps = ["a"]

for wp in wps:
    net.build_wp(wp)

net.init_weights(sess)


def generate_random_input():
    input = np.random.randint(0, 100, (BS, TS, F))
    labels = np.mean(input, 2)
    mask = np.ones((BS, TS), np.float32)
    seq_len = np.full((BS,), TS).reshape((-1,))
    return input, labels, mask, seq_len


# for wp in wps:
#     for epoch in range(EPOCHS):
#         input, labels, mask, seq_len = generate_random_input()
#         state = net.zero_wp_state(wp, BS)
#         state, sse, returns = net.fit_wp(sess, wp, state, input, labels, mask, seq_len)
#         print("%s %d %.2f" % (wp, epoch, sse))
#     net.save_wp(sess, wp, EPOCHS)
#
# net.save_graph(graph)

for wp in wps:
    net.load_wp(sess, wp, EPOCHS)
    input, labels, mask, seq_len = generate_random_input()
    state = net.zero_wp_state(wp, BS)
    new_state, sse, returns = net.eval_wp(sess, wp, state, input, labels, mask, seq_len)
    print("Eval %s %.2f" % (wp, sse))