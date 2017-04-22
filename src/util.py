from six.moves import cPickle as pickle
import numpy as np
import os
import unittest
import time
import matplotlib.pyplot as plt


def plot9images(images, cls_true, img_shape, cls_pred=None):
    """
    Function to show 9 images with their respective classes.
    If cls_pred is an array, you can see the image and the prediction.

    :type images: np array
    :type cls_true: np array
    :type img_shape: np array
    :type cls_prediction: None or np array
    """
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def randomize_in_place(list1, list2, init):
    """
    Function to randomize two lists the same way.
    Usualy this functions is used when list1 = dataset,
    and list2 = labels.

    :type list1: list
    :type list2: list
    :type init: int
    """
    np.random.seed(seed=init)
    np.random.shuffle(list1)
    np.random.seed(seed=init)
    np.random.shuffle(list2)


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.

    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def get_log_path():
    """
    Function to create one unique path for each model.
    This path is created by using the specific time that
    the function is called.

    :rtype: str
    """
    log_basedir = 'logs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')
    log_path = os.path.join(log_basedir, run_label)
    return log_path


def get_data():
    """
    Function to get all the datasets from the pickle in

    https://www.dropbox.com/s/t5l172b417p9pf1/notMNIST.zip?dl=0

    it is require that the pickle is downloaded.
    The scrip download.sh should be run before call this function

    :rtype train_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_labels: np arrays
    :rtype test_dataset: np arrays
    :rtype test_labels: np arrays
    """
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


def reformat_4d(dataset, labels, image_size=28, num_channels=1, num_labels=10):
    """
    Function that reformats the dataset to be a 4 dimension
    and transorm the labels array to be one array of one-hot vectors.

    :type dataset: np arrays
    :type labels: np arrays
    :type image_size: int
    :type num_channel: int
    :type num_labels: int
    :rtype dataset: np arrays
    :rtype labels: np arrays
    """
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def get_data_4d():
    """
    Similar as the get_data function. The only difference
    is that all the arrays are transfor by the reformat_4d function.

    :rtype train_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_dataset: np arrays
    :rtype train_labels: np arrays
    :rtype valid_labels: np arrays
    :rtype test_dataset: np arrays
    :rtype test_labels: np arrays
    """
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_data()
    train_dataset, train_labels = reformat_4d(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat_4d(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat_4d(test_dataset, test_labels)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
