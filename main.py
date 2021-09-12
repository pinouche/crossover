import numpy as np
from timeit import default_timer as timer
import warnings
import pickle
import copy

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import keras

from load_data import load_cifar
from load_data import load_mnist
from load_data import load_cifar_100

from utils import identify_interesting_neurons
from utils import transplant_neurons
from utils import get_hidden_layers
from utils import match_random_filters
from utils import get_corr_cnn_filters
from utils import crossover_method
from utils import arithmetic_crossover

from neural_models import keras_model_cnn
warnings.filterwarnings("ignore")


def transplant_crossover(crossover, data, x_train, y_train, x_test, y_test, num_transplants, num_trainable_layer=5, batch_size_activation=512,
                         batch_size_sgd=128, work_id=0):

    result_list = []
    print("crossover method: " + crossover)
    for safety_level in ["safe_crossover", "naive_crossover"]:
        print(safety_level)

        loss_list = []
        for epoch in range(num_transplants + 1):
            print("Transplant number: " + str(epoch))

            # reset upper layers to random initialization
            model_offspring_one = keras_model_cnn(work_id, data)
            model_offspring_two = keras_model_cnn(work_id + 1000, data)

            if epoch > 0:
                # get the randomly reset weights
                model_offspring_one.set_weights(weights_offspring_one)
                model_offspring_two.set_weights(weights_offspring_two)

                loss_after_transplant_one = model_offspring_one.evaluate(x_test, y_test)[0]
                loss_after_transplant_two = model_offspring_two.evaluate(x_test, y_test)[0]

            model_information_offspring_one = model_offspring_one.fit(x_train, y_train, batch_size=batch_size_sgd, epochs=50,
                                                                      verbose=2, validation_data=(x_test, y_test))

            model_information_offspring_two = model_offspring_two.fit(x_train, y_train, batch_size=batch_size_sgd, epochs=50,
                                                                      verbose=2, validation_data=(x_test, y_test))

            loss_one = model_information_offspring_one.history["val_loss"]
            loss_two = model_information_offspring_two.history["val_loss"]

            if epoch > 0:
                loss_one.insert(0, loss_after_transplant_one)
                loss_two.insert(0, loss_after_transplant_two)

            loss_list.append(loss_one)
            loss_list.append(loss_two)

            weights_offspring_one = model_offspring_one.get_weights()
            weights_offspring_two = model_offspring_two.get_weights()

            # compute the cross correlation matrix
            hidden_representation_offspring_one = get_hidden_layers(model_offspring_one, x_test, batch_size_activation)
            hidden_representation_offspring_two = get_hidden_layers(model_offspring_two, x_test, batch_size_activation)

            list_cross_corr = get_corr_cnn_filters(hidden_representation_offspring_one, hidden_representation_offspring_two)
            self_corr_offspring_one = get_corr_cnn_filters(hidden_representation_offspring_one, hidden_representation_offspring_one)
            self_corr_offspring_two = get_corr_cnn_filters(hidden_representation_offspring_two, hidden_representation_offspring_two)

            # functionally align the networks
            list_ordered_indices_one, list_ordered_indices_two, weights_offspring_one, weights_offspring_two = crossover_method(
                weights_offspring_one, weights_offspring_two, list_cross_corr, safety_level)

            # re-order the correlation matrices
            list_cross_corr = [list_cross_corr[index][:, list_ordered_indices_two[index]] for index in
                               range(len(list_ordered_indices_two))]

            q_values_list = [0.5] * len(list_cross_corr)

            if crossover == "targeted_crossover_low_corr":
                # identify neurons to transplant from offspring two to offspring one
                list_neurons_to_transplant_one, list_neurons_to_remove_one = identify_interesting_neurons(list_cross_corr,
                                                                                                          self_corr_offspring_one,
                                                                                                          self_corr_offspring_two)

                # identify neurons to transplant from offspring one to offspring two
                list_cross_corr_transpose = [np.transpose(corr_matrix) for corr_matrix in list_cross_corr]
                list_neurons_to_transplant_two, list_neurons_to_remove_two = identify_interesting_neurons(list_cross_corr_transpose,
                                                                                                          self_corr_offspring_two,
                                                                                                          self_corr_offspring_one)

            elif crossover == "targeted_crossover_random":
                list_neurons_to_transplant_one, list_neurons_to_remove_one = match_random_filters(q_values_list, list_cross_corr)
                list_neurons_to_transplant_two, list_neurons_to_remove_two = match_random_filters(q_values_list, list_cross_corr)

            if crossover == "arithmetic_crossover":
                weights_offspring_one = arithmetic_crossover(weights_offspring_one, weights_offspring_two)

                weights_offspring_one_copy = copy.deepcopy(weights_offspring_one)
                weights_offspring_two = weights_offspring_one_copy

            else:

                weights_offspring_one_tmp = copy.deepcopy(weights_offspring_one)
                weights_offspring_two_tmp = copy.deepcopy(weights_offspring_two)

                depth = 0
                for layer in range(num_trainable_layer - 1):
                    # transplant offspring one
                    weights_offspring_one = transplant_neurons(weights_offspring_one, weights_offspring_two_tmp, list_neurons_to_transplant_one,
                                                               list_neurons_to_remove_one, layer, depth)

                    # transplant offspring two
                    weights_offspring_two = transplant_neurons(weights_offspring_two, weights_offspring_one_tmp, list_neurons_to_transplant_two,
                                                               list_neurons_to_remove_two, layer, depth)

                    depth = (layer + 1) * 6

        result_list.append(loss_list)

        keras.backend.clear_session()

    return result_list


def average_weights_crossover(crossover, data, x_train, y_train, x_test, y_test, num_transplants, batch_size_activation=512,
                         batch_size_sgd=128, work_id=0):
    result_list = []
    print("crossover method: " + crossover)

    for epoch in range(num_transplants + 1):
        print("Transplant number: " + str(epoch))

        # reset upper layers to random initialization
        model_parent = keras_model_cnn(work_id, data)
        model_parent.fit(x_train, y_train, batch_size=batch_size_sgd, epochs=50,
                                                                      verbose=2, validation_data=(x_test, y_test))

        weights_parent = model_parent.get_weights()

        model_parent_one = keras_model_cnn(0, data)
        model_parent_two = keras_model_cnn(0, data)
        model_parent_one.set_weights(weights_parent)
        model_parent_two.set_weights(weights_parent)

        model_parent_one_info = model_parent_one.fit(x_train, y_train, batch_size=batch_size_sgd, epochs=10,
                                        verbose=2, validation_data=(x_test, y_test))
        model_parent_two_info = model_parent_two.fit(x_train, y_train, batch_size=batch_size_sgd, epochs=10,
                                        verbose=2, validation_data=(x_test, y_test))

        best_parent_loss = min(model_parent_one_info.history["val_loss"][-1], model_parent_two_info.history["val_loss"][-1])

        weights_parent_one = model_parent_one.get_weights()
        weights_parent_two = model_parent_two.get_weights()

        # compute the cross correlation matrix
        hidden_representation_offspring_one = get_hidden_layers(model_parent_one, x_test, batch_size_activation)
        hidden_representation_offspring_two = get_hidden_layers(model_parent_two, x_test, batch_size_activation)

        list_cross_corr = get_corr_cnn_filters(hidden_representation_offspring_one, hidden_representation_offspring_two)

        for safety_level in ["safe_crossover", "naive_crossover"]:
            weights_parent_one_copy = copy.deepcopy(weights_parent_one)
            weights_parent_two_copy = copy.deepcopy(weights_parent_two)

            # functionally align the networks
            list_ordered_indices_one, list_ordered_indices_two, weights_offspring_one, weights_offspring_two = crossover_method(
                weights_parent_one_copy, weights_parent_two_copy, list_cross_corr, safety_level)

            weights_offspring = arithmetic_crossover(weights_offspring_one, weights_offspring_two)
            model_offspring = keras_model_cnn(0, data)
            model_offspring.set_weights(weights_offspring)

            loss_after_crossover = model_offspring.evaluate(x_test, y_test)[0]

            improvement = ((loss_after_crossover - best_parent_loss) / best_parent_loss) * -100
            result_list.append(improvement)
            print("IMPROVEMENT: ", safety_level, improvement)

    keras.backend.clear_session()

    return result_list


def crossover_offspring(data, x_train, y_train, x_test, y_test, work_id=0):
    # shuffle input data here

    np.random.seed(work_id + 1)
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    # program hyperparameters
    num_trainable_layer = 5
    batch_size_activation = 512  # batch_size to compute the activation maps
    batch_size_sgd = 128

    num_transplants = 1

    # crossover = "targeted_crossover_low_corr"
    # crossover = "targeted_crossover_random"
    crossover = "arithmetic_crossover"

    if crossover == "transplant_crossover":
        result_list = transplant_crossover(crossover, data, x_train, y_train, x_test, y_test, num_transplants, num_trainable_layer,
                                           batch_size_activation, batch_size_sgd, work_id)
    else:
        result_list = average_weights_crossover(crossover, data, x_train, y_train, x_test, y_test, num_transplants, batch_size_activation,
                                                batch_size_sgd, work_id)

    return result_list


if __name__ == "__main__":
    
    data = "cifar10"

    if data == "cifar10":
        x_train, x_test, y_train, y_test = load_cifar()
    elif data == "cifar100":
        x_train, x_test, y_train, y_test = load_cifar_100()
    elif data == "mnist":
        x_train, x_test, y_train, y_test = load_mnist()

    num_processes = 1

    start = timer()

    pair_list = [pair for pair in range(num_processes)]

    results = crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list)

    pickle.dump(results, open("crossover_results.pickle", "wb"))

    end = timer()
    print(end - start)
