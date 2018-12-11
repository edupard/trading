import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import copy


class RnnNetConfig(object):
    FEATURES_DIM = 1
    LSTM_LAYERS = 3
    LSTM_LAYER_SIZE = 30
    FC_LAYERS = [100]
    DROPOUT_KEEP_RATE = 1.0


_config_proto = RnnNetConfig()


def defaultRnnNetConfig():
    return copy.deepcopy(_config_proto)


class RnnNet(object):

    def __init__(self, config: RnnNetConfig, graph: tf.Graph):
        self.config = config
        self.graph = graph
        print('Creating model averaging network input...')
        with graph.as_default():
            with tf.variable_scope('input') as scope:
                self.input = tf.placeholder(tf.float32, [None, None, self.config.FEATURES_DIM], name='input')
                self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
                self.labels = tf.placeholder(tf.float32, [None, None], name='labels')
                self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
                self.keep_prob = tf.placeholder(tf.float32)
        self.state = {}
        self.new_state = {}
        self.rnn_cell = {}
        self.returns = {}
        self.sse = {}
        self.cost = {}
        self.optimizer = {}
        self.vars = {}
        self.grads_and_vars = {}
        self.saver = {}
        self.train = {}

    def build_wp(self, id):
        print('creating %s weak predictor network...' % id)
        with self.graph.as_default() as graph:
            with tf.variable_scope(id) as scope:

                self.rnn_cell[id] = rnn_cell = rnn.MultiRNNCell(
                    [
                        rnn.DropoutWrapper(rnn.LSTMCell(self.config.LSTM_LAYER_SIZE), input_keep_prob=self.keep_prob)
                        for _ in range(self.config.LSTM_LAYERS)
                    ])

                state = ()
                for s in rnn_cell.state_size:
                    c = tf.placeholder(tf.float32, [None, s.c])
                    h = tf.placeholder(tf.float32, [None, s.h])
                    state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
                self.state[id] = state

                # Batch size x time steps x features.
                output, new_state = tf.nn.dynamic_rnn(rnn_cell, self.input, initial_state=state, sequence_length=self.seq_len)
                self.new_state[id] = new_state

                fc_layer_idx = 0
                for num_units in self.config.FC_LAYERS:
                    scope_name = 'fc_layer_%d' % fc_layer_idx
                    with tf.name_scope(scope_name):
                        output = tf.contrib.layers.fully_connected(output, num_units, activation_fn=tf.nn.relu,
                                                                   scope='dense_%d' % fc_layer_idx)
                        output = tf.nn.dropout(output, self.keep_prob)
                    fc_layer_idx += 1

                # final layer to make prediction
                with tf.name_scope('prediction_layer'):
                    self.returns[id] = returns = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

                with tf.name_scope('loss'):
                    diff = returns - tf.expand_dims(self.labels, 2)
                    self.sse[id] = sse = tf.reduce_sum(tf.multiply(tf.square(diff), tf.expand_dims(self.mask, 2)))
                    self.cost[id] = cost = sse / tf.reduce_sum(self.mask)

                    self.optimizer[id] = optimizer = tf.train.AdamOptimizer()
                    self.vars[id] = vars = tf.trainable_variables(scope.name)
                    self.grads_and_vars[id] = grads_and_vars = optimizer.compute_gradients(cost, var_list=vars)
                    self.train[id] = optimizer.apply_gradients(grads_and_vars)

                self.saver[id] = tf.train.Saver(tf.trainable_variables(scope.name), max_to_keep=None)

    def init_weights(self, sess: tf.Session):
        print('initializing weights...')
        with self.graph.as_default() as graph:
            init = tf.global_variables_initializer()
            sess.run(init)

    def zero_wp_state(self, id, batch_size):
        zero_state = ()
        for s in self.rnn_cell[id].state_size:
            c = np.zeros((batch_size, s.c))
            h = np.zeros((batch_size, s.h))
            zero_state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
        return zero_state

    def _fill_feed_dict(self, id, feed_dict, state):
        idx = 0
        for s in state:
            feed_dict[self.state[id][idx].c] = s.c
            feed_dict[self.state[id][idx].h] = s.h
            idx += 1

    def eval_wp(self, sess, id, state, input, labels, mask, seq_len):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask, self.keep_prob: 1.0,
                     self.seq_len: seq_len}
        self._fill_feed_dict(id, feed_dict, state)

        new_state, sse, returns = sess.run((self.new_state[id], self.sse[id], self.returns[id]), feed_dict)
        return new_state, sse, returns

    def fit_wp(self, sess, id, state, input, labels, mask, seq_len):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask,
                     self.keep_prob: self.config.DROPOUT_KEEP_RATE, self.seq_len: seq_len}
        self._fill_feed_dict(id, feed_dict, state)
        new_state, sse, returns, _ = sess.run((self.new_state[id], self.sse[id], self.returns[id], self.train[id]), feed_dict)
        return new_state, sse, returns

    def save_wp(self, sess, id, epoch):
        print('saving %s %d epoch weights' % (id, epoch))
        self.saver[id].save(sess, './weights/%s' % id, global_step=epoch, write_meta_graph=False)

    def load_wp(self, sess, id, epoch):
        print('loading %d epoch weights' % epoch)
        self.saver[id].restore(sess, "./weights/%s-%d" % (id, epoch))

    def save_graph(self, graph):
        print('saving computation graph')
        tf.summary.FileWriter('./graph',
                              graph)


