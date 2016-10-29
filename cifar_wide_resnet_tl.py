import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np


class CNNEnv:
    def __init__(self):

        # The data, shuffled and split between train and test sets
        self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

        # Reorder dimensions for tensorflow
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # For wide resnets
        self.blocks_per_group = 4
        self.widening_factor = 4

        # Basic info
        self.batch_num = 64
        self.img_row = 32
        self.img_col = 32
        self.img_channels = 3
        self.nb_classes = 10

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]

    def reset(self, first):
        self.first = first
        if self.first is True:
            self.sess.close()
        self.sess = tf.InteractiveSession()

    def step(self):

        def zero_pad_channels(x, pad=0):
            """
            Function for Lambda layer
            """
            pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
            return tf.pad(x, pattern)

        def residual_block(x, nb_filters=16, subsample_factor=1):
            prev_nb_channels = x.get_shape()[3]
            # shape = x.get_shape()
            # prev_nb_channels = tuple([i.__int__() for i in shape])

            if subsample_factor > 1:
                subsample = [1, subsample_factor, subsample_factor, 1]
                # shortcut: subsample + zero-pad channel dim
                shortcut = tl.layers.PoolLayer(x,
                                               ksize=subsample,
                                               strides=subsample,
                                               padding='VALID',
                                               pool=tf.nn.avg_pool)

            else:
                subsample = [1, 1, 1, 1]
                # shortcut: identity
                shortcut = x

            if nb_filters > prev_nb_channels:
                shortcut = tl.layers.LambdaLayer(shortcut, zero_pad_channels, arguments={'pad': nb_filters - prev_nb_channels})

            y = tl.layers.BatchNormLayer(x, decay=0.999, epsilon=1e-05, is_train=True)
            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, 3, nb_filters],
                                      strides=subsample,
                                      padding='SAME')

            y = tl.layers.BatchNormLayer(y, decay=0.999, epsilon=1e-05,
                                         is_train=True)

            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, 3, nb_filters],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')

            out = tf.layers.ElementwiseLayer([y, shortcut], combine_fn=tf.add)

            class LambdaLayer(Layer):
                def __init__(
                        self,
                        layer=None,
                        fn=None,
                        name='lambda_layer',
                ):
                    Layer.__init__(self, name=name)
                    self.inputs = layer.outputs

                    print(
                    "  tensorlayer:Instantiate LambdaLayer  %s" % self.name)
                    with tf.variable_scope(name) as vs:
                        self.outputs = fn(self.inputs)

                    self.all_layers = list(layer.all_layers)
                    self.all_params = list(layer.all_params)
                    self.all_drop = dict(layer.all_drop)
                    self.all_layers.extend([self.outputs])

            return out

        # Placeholders
        learning_rate = tf.placeholder(tf.float32)
        img = tf.placeholder(tf.float32, shape=[self.batch_num, 32, 32, 3])
        labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

        x = tl.layers.InputLayer(img)
        x = tl.layers.Conv2dLayer(x, act=tf.nn.relu, shape=[3, 3, 3, 16], strides=[1, 1, 1, 1], padding='SAME')

        for i in range(0, self.blocks_per_group):
            nb_filters = 16 * self.widening_factor
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

        for i in range(0, self.blocks_per_group):
            nb_filters = 32 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        for i in range(0, self.blocks_per_group):
            nb_filters = 64 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        x = tl.layers.BatchNormLayer(x, decay=0.999, epsilon=1e-05, is_train=True)

        x = tl.layers.PoolLayer(x,
                                ksize=[1, 8, 8, 1],
                                strides=[1, 8, 8, 1],
                                padding='VALID',
                                pool=tf.nn.avg_pool)

        x = tl.layers.FlattenLayer(x)

        x = tl.layers.DenseLayer(x, n_units=self.nb_classes, act=tf.identity)

        output = x.outputs

        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels))
        cost = ce

        correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), labels)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_params = x.all_params
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate, use_locking=False).minimize(cost, var_list=train_params)

        self.sess.run(tf.initialize_all_variables())

        for i in range(10):
            batch = self.next_batch(self.batch_num)
            feed_dict = {img: batch[0], labels: batch[1], learning_rate: 0.01}
            feed_dict.update(x.all_drop)
            _, l, ac = self.sess.run([train_op, cost, acc], feed_dict=feed_dict)

        '''
        with sess.as_default():

            for i in range(10):

                batch = self.next_batch(self.batch_num)
                _, l = sess.run([optimizer, loss],
                                feed_dict={img: batch[0], labels: batch[1]})
                print(l)
        '''

        '''
        with sess.as_default():
            acc = acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})
            print(acc)
        '''

a = CNNEnv()
a.reset(first=False)
a.step()