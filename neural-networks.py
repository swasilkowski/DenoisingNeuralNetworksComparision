import tensorflow as tf
from functools import partial

class AutoEncoder:

    

    def __init__(self, n_inputs, training_set):
        n_hidden1 = n_inputs / 2
        n_hidden2 = n_inputs / 4
        n_hidden3 = n_hidden1
        n_outputs = n_inputs

        learning_rate = 0.01
        l2_reg = 0.0001

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        he_init = tf.contrib.layers.variance_scaling_initializer()
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

        hidden1 = my_dense_layer(X, n_hidden1)
        hidden2 = my_dense_layer(hidden1, n_hidden2)
        hidden3 = my_dense_layer(hidden2, n_hidden3)
        self.outputs = tf.layers.dense(hidden3, n_outputs, activation=None, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

        reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        n_epochs = 5

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                sess.run(training_op, feed_dict={X: training_set})
            save_path = self.saver.save(sess, "./autoencoder_model.ckpt")


    def test(self, test_set):
        with tf.Session() as sess:
            self.saver.restore(sess, "./autoencoder_model.ckpt")
            output_val = self.outputs.eval(feed_dict={X: test_set})