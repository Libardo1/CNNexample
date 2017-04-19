from six.moves import cPickle as pickle
import numpy as np
import os
import unittest
import time


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.
    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def accuracy(predictions, labels):
    comparisson = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return (100.0 * comparisson / predictions.shape[0])


def get_log():
    log_basedir = 'logs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')
    log_path = os.path.join(log_basedir, run_label)
    return log_path


def get_data():
    currentdir = os.path.dirname(__file__)
    filepath = os.path.join(currentdir, "data")
    filepath = os.path.join(filepath, "notMNIST.pickle")
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


# reformating the dataset to be a 4 dimension tensor
def reformat_4d(dataset, labels, image_size=28, num_channels=1, num_labels=10):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def get_data_4d():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data()
    train_dataset, train_labels = reformat_4d(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat_4d(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat_4d(test_dataset, test_labels)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
