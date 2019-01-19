import tensorflow as tf

from functools import partial


class AutoEncoder:
    def __init__(self, n_inputs, training_set_X=[], training_set_Y=[], batch_size=1, n_epochs=1):
        n_batches = len(training_set_X)//batch_size
        n_hidden1 = n_inputs / 2
        n_hidden2 = n_inputs / 4
        n_hidden3 = n_hidden1
        n_outputs = n_inputs

        learning_rate = 0.01
        l2_reg = 0.0001

        self.X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.Y = tf.placeholder(tf.float32, shape=[None, n_outputs])

        he_init = tf.contrib.layers.variance_scaling_initializer()
        l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

        hidden1 = my_dense_layer(self.X, n_hidden1)
        hidden2 = my_dense_layer(hidden1, n_hidden2)
        hidden3 = my_dense_layer(hidden2, n_hidden3)
        self.outputs = tf.layers.dense(hidden3, n_outputs, activation=None, kernel_initializer=he_init, kernel_regularizer=l2_regularizer)

        reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.Y))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        if(training_set_X.any()):
            with tf.Session() as sess:
                init.run()
                for epoch in range(n_epochs):
                    for batch_n in range(n_batches):
                        batch_X = [training_set_X[i] for i in range(batch_n * batch_size, (batch_n + 1) * batch_size - 1)]
                        batch_Y = [training_set_Y[i] for i in range(batch_n * batch_size, (batch_n + 1) * batch_size - 1)]
                        sess.run(training_op, feed_dict={self.X: batch_X, self.Y: batch_Y})
                    print("Epoch " + str(epoch) + " done")
                    
                save_path = self.saver.save(sess, "./autoencoder_model.ckpt")


    def test(self, test_set):
        with tf.Session() as sess:
            self.saver.restore(sess, "./autoencoder_model.ckpt")
            output_val = self.outputs.eval(feed_dict={self.X: test_set})
            return output_val