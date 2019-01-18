import tensorflow as tf
import numpy as np

import xrbm.models
import xrbm.train
import xrbm.losses


class RBM:
    def __init__(self, window, trainX, hid_coef, epochs, train_rate, batch_size):
        num_vis         = window
        num_hid         = hid_coef * window
        learning_rate   = train_rate
        training_epochs = epochs
        batchsize       = batch_size

        tf.reset_default_graph()
        self.rbm = xrbm.models.RBM(num_vis=num_vis, num_hid=num_hid, name='denoise_rbm')

        batch_idxs     = np.random.permutation(range(len(trainX)))
        n_batches      = len(batch_idxs) // batch_size

        self.features_placeholder = tf.placeholder(tf.float32, shape=(None, window))
        momentum = tf.placeholder(tf.float32, shape=())

        cdapproximator = xrbm.train.CDApproximator(learning_rate=learning_rate)
        self.train_op = cdapproximator.train(self.rbm, vis_data=self.features_placeholder)
        
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):

            for batch_i in range(n_batches):
                idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
                self.sess.run(self.train_op, feed_dict={self.features_placeholder: trainX[idxs_i]})
            print("Epoch " + str(epoch) + " done")
        save_path = self.saver.save(self.sess, "./rbm_model.ckpt")


    def test(self, testX):
        #validation_placeholder = tf.placeholder(tf.float32, testX.shape)
        testY = self.sess.run(self.train_op, feed_dict={self.features_placeholder: testX})

        return testY