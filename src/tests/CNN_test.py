import unittest
import os
import sys
import tensorflow as tf
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from util import run_test, get_data_4d
from CNN import Config, CNNModel, DataHolder


class TestCNN(unittest.TestCase):
    """
    Class that test the CNN model
    """
    @classmethod
    def setUpClass(cls):
        cls.train_dataset, cls.train_labels, cls.valid_dataset, cls.valid_labels, cls.test_dataset, cls.test_labels = get_data_4d()
        dataholder = DataHolder(cls.train_dataset,
                                cls.train_labels,
                                cls.valid_dataset,
                                cls.valid_labels,
                                cls.test_dataset,
                                cls.test_labels)
        config = Config()
        cls.model = CNNModel(config, dataholder)

    def test_forward(self):
        """
        Test to check if the forward propagation is working.
        """
        batch_size = TestCNN.model.batch_size
        with tf.Session(graph=TestCNN.model.graph) as session:
            tf.global_variables_initializer().run()
            offset = batch_size % (TestCNN.train_labels.shape[0] - batch_size)
            batch_data = TestCNN.train_dataset[offset:(offset + batch_size), :]
            batch_labels = TestCNN.train_labels[offset:(offset + batch_size), :]
            feed_dict = {TestCNN.model.tf_train_dataset: batch_data, TestCNN.model.tf_train_labels: batch_labels}
            loss = session.run(TestCNN.model.loss, feed_dict=feed_dict)
        self.assertTrue(loss < 3)

    def test_opt(self):
        """
        Test to check if the optimazation is working.
        """
        num_steps = 101
        batch_size = TestCNN.model.batch_size
        with tf.Session(graph=TestCNN.model.graph) as session:
            tf.global_variables_initializer().run()
            for step in range(num_steps):
                offset = (step * batch_size) % (TestCNN.train_labels.shape[0] - batch_size)
                batch_data = TestCNN.train_dataset[offset:(offset + batch_size), :]
                batch_labels = TestCNN.train_labels[offset:(offset + batch_size), :]

                feed_dict = {TestCNN.model.tf_train_dataset: batch_data,
                             TestCNN.model.tf_train_labels: batch_labels}

                _, current_loss = session.run([TestCNN.model.optimizer,
                                               TestCNN.model.loss],
                                              feed_dict=feed_dict)
                if step == 0:
                    initial_loss = current_loss
                if step == 100:
                    final_loss = current_loss
        self.assertTrue(final_loss < initial_loss)


if __name__ == "__main__":
    run_test(TestCNN,
             "\n=== Running CNN tests ===\n")
