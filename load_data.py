import keras


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, x_test, y_train, y_test


def load_cifar_100(label_mode="fine"):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode)
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