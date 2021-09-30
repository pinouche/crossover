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


def add_negative_class_examples(data_main, data_subset, list_classes, train_data=True):
    index1, index2 = 0, 1
    if not train_data:
        index1, index2 = 2, 3

    num_instances_main = data_main[index1].shape[0]
    num_instances_subset = data_subset[index1].shape[0]

    indices = np.array(range(num_instances_main))
    chosen_indices = np.random.choice(indices, num_instances_subset, replace=False)

    data_subset[index1] = np.concatenate([data_subset[index1], data_main[index1][chosen_indices]])
    data_subset[index2] = np.concatenate([data_subset[index2], data_main[index2][chosen_indices]])
    data_subset[index2][[v not in list_classes for v in data_subset[index2]]] = 1

    return data_subset


def shift_labels(label_train, label_test):

    label_train = np.array([list(np.unique(label_train)).index(element) for element in label_train])
    label_test = np.array([list(np.unique(label_test)).index(element) for element in label_test])

    return label_train, label_test
