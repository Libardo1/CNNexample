from six.moves import cPickle as pickle
import numpy as np
import os
import unittest


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.
    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


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

def main():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data_4d()
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

if __name__ == "__main__":
    main()