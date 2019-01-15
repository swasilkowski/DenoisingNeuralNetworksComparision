import tensorflow as tf

import xrbm.models
import xrbm.train
import xrbm.losses


class RBM:
    def __init__(self, window, trainX):
        num_vis         = window
        num_hid         = 0.4 * window
        learning_rate   = 0.1
        training_epochs = 50

        tf.reset_default_graph()
        self.rbm = xrbm.models.RBM(num_vis=num_vis, num_hid=num_hid, name='denoise_rbm')

        self.features_placeholder = tf.placeholder(tf.float32, shape=(None, window))

        cdapproximator = xrbm.train.CDApproximator(learning_rate=learning_rate)
        self.train_op = cdapproximator.train(self.rbm, vis_data=self.features_placeholder)
        
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            self.sess.run(self.train_op, feed_dict={self.features_placeholder: trainX})
            print("Epoch " + str(epoch) + " done")
        save_path = self.saver.save(self.sess, "./rbm_model.ckpt")


    def test(self, testX):
        #validation_placeholder = tf.placeholder(tf.float32, testX.shape)
        testY = self.sess.run(self.train_op, feed_dict={self.features_placeholder: testX})

        return testY