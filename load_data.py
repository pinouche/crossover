import keras
import numpy as np


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, x_test, y_train, y_test


def load_cifar_100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data("coarse")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, x_test, y_train, y_test


def load_cifar():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, x_test, y_train, y_test


def subset_classes(y_train, y_test, class_subset_list=[]):

    if len(class_subset_list) != 0:
        mask_train = [value in class_subset_list for value in y_train]
        mask_test = [value in class_subset_list for value in y_test]

    else:
        mask_train = [True] * y_train.shape[0]
        mask_test = [True] * y_test.shape[0]

    return np.array(mask_train), np.array(mask_test)


def shift_labels(label_train, label_test):

    label_train = np.array([list(np.unique(label_train)).index(element) for element in label_train])
    label_test = np.array([list(np.unique(label_test)).index(element) for element in label_test])

    return label_train, label_test
