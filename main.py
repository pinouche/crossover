import numpy as np
from timeit import default_timer as timer
import multiprocessing
import warnings
import pickle
import os
import copy

from keras.models import load_model
import keras

from utils import load_cifar
from utils import load_mnist
from utils import load_cifar_100
from utils import partition_classes
from utils import get_fittest_network

from utils import compute_neurons_variance
from utils import identify_interesting_neurons
from utils import transplant_neurons
from utils import get_hidden_layers
from utils import match_random_filters
from utils import get_corr_cnn_filters
from utils import crossover_method
from utils import compute_q_values
from utils import mean_ensemble
from utils import reset_weights_layer

from neural_models import CustomSaver
from neural_models import lr_scheduler
from neural_models import keras_model_cnn
from neural_models import keras_vgg

warnings.filterwarnings("ignore")


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc):
    # shuffle input data here

    num_pairs = len(pair_list)
    print("FOR PAIR NUMBER " + str(work_id))

    np.random.seed(work_id + 1)
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    # program hyperparameters
    num_trainable_layer = 7
    mix_full_networks = True
    total_training_epoch = 20 * num_trainable_layer
    batch_size_activation = 2048  # batch_size to compute the activation maps
    batch_size_sgd = 512
    cut_off = 0.2
    result_list = []

    if not mix_full_networks:
        x_train_s1, y_train_s1, x_test_s1, y_test_s1, x_train_s2, y_train_s2, x_test_s2, y_test_s2 = \
            partition_classes(x_train, x_test, y_train, y_test, cut_off)

    if mix_full_networks:
        crossover_types = ["frozen_aligned_targeted_crossover_low_corr", "frozen_aligned_targeted_crossover_random"]
    else:
        crossover_types = ["aligned_targeted_crossover_low_corr", "aligned_targeted_crossover_random",
                           "large_subset_fine_tune"]

    for crossover in crossover_types:

        print("crossover method: " + crossover)

        if crossover in ["frozen_aligned_targeted_crossover_low_corr", "frozen_aligned_targeted_crossover_random"]:

            q_values_list = [0] * (num_trainable_layer - 1)

            for safety_level in ["safe_crossover", "naive_crossover"]:

                print(safety_level)

                loss_list = []

                for layer in range(num_trainable_layer - 1):

                    print("the layer number is: " + str(layer))

                    model_offspring_one = keras_model_cnn(work_id + num_pairs, data)
                    model_offspring_two = keras_model_cnn(work_id + (2 * num_pairs), data)

                    if layer > 0:
                        weights_offspring_one = reset_weights_layer(weights_offspring_one, layer)
                        weights_offspring_two = reset_weights_layer(weights_offspring_two, layer)
                        model_offspring_one.set_weights(weights_offspring_one)
                        model_offspring_two.set_weights(weights_offspring_two)

                    model_information_offspring_one = model_offspring_one.fit(x_train, y_train,
                                                                              batch_size=batch_size_sgd,
                                                                              epochs=int(
                                                                                  total_training_epoch / num_trainable_layer),
                                                                              verbose=2,
                                                                              validation_data=(x_test, y_test))

                    model_information_offspring_two = model_offspring_two.fit(x_train, y_train,
                                                                              batch_size=batch_size_sgd,
                                                                              epochs=int(
                                                                                  total_training_epoch / num_trainable_layer),
                                                                              verbose=2,
                                                                              validation_data=(x_test, y_test))

                    loss_list.append(model_information_offspring_one.history["val_loss"])
                    loss_list.append(model_information_offspring_two.history["val_loss"])

                    weights_offspring_one = model_offspring_one.get_weights()
                    weights_offspring_two = model_offspring_two.get_weights()

                    model_offspring_one.set_weights(weights_offspring_one)
                    model_offspring_two.set_weights(weights_offspring_two)

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

                    # compute the q values for each layer (we use the same q values for naive alignment too)
                    if safety_level == "safe_crossover":
                        q_values_list[layer:] = compute_q_values(list_cross_corr)[layer:]

                        if not mix_full_networks:
                            q_values_list = [cut_off] * len(list_cross_corr)

                    # identify neurons to transplant from offspring two to offspring one
                    list_neurons_to_transplant_one, list_neurons_to_remove_one = identify_interesting_neurons(list_cross_corr,
                                                                                                              self_corr_offspring_one,
                                                                                                              q_values_list)

                    # identify neurons to transplant from offspring one to offspring two
                    list_cross_corr_transpose = [np.transpose(corr_matrix) for corr_matrix in list_cross_corr]
                    list_neurons_to_transplant_two, list_neurons_to_remove_two = identify_interesting_neurons(list_cross_corr_transpose,
                                                                                                              self_corr_offspring_two,
                                                                                                              q_values_list)

                    if crossover == "frozen_aligned_targeted_crossover_random":
                        list_neurons_to_transplant_one, list_neurons_to_remove_one = match_random_filters(q_values_list, list_cross_corr)
                        list_neurons_to_transplant_two, list_neurons_to_remove_two = match_random_filters(q_values_list, list_cross_corr)

                    weights_offspring_one_tmp = copy.deepcopy(weights_offspring_one)
                    weights_offspring_two_tmp = copy.deepcopy(weights_offspring_two)

                    depth = 0
                    for l in range(layer):

                        # transplant offspring one
                        weights_offspring_one = transplant_neurons(weights_offspring_one, weights_offspring_two_tmp, list_neurons_to_transplant_one,
                                                                   list_neurons_to_remove_one, l, depth)

                        # transplant offspring one
                        weights_offspring_two = transplant_neurons(weights_offspring_two, weights_offspring_one_tmp, list_neurons_to_transplant_two,
                                                                   list_neurons_to_remove_two, l, depth)

                        depth = (layer + 1) * 6

                        list_ordered_indices_one, list_ordered_indices_two, weights_offspring_one, weights_offspring_two = crossover_method(
                            weights_offspring_one,
                            weights_offspring_two,
                            list_cross_corr,
                            safety_level)

                        # update the cross-correlation matrix
                        list_cross_corr = [list_cross_corr[index][:, list_ordered_indices_two[index]] for index in
                                           range(len(list_ordered_indices_two))]

                loss_list = [val for sublist in loss_list for val in sublist]
                result_list.append(loss_list)

        elif crossover == "large_subset_fine_tune":
            different_safety_weights_list = []
            fittest_weights = copy.deepcopy(weights_nn_one)
            different_safety_weights_list.append(fittest_weights)

        elif crossover == "mean_ensemble":
            loss = mean_ensemble(model_one, model_two, x_test, y_test)

            result_list.append(loss)

        if crossover in ["aligned_targeted_crossover_low_corr", "aligned_targeted_crossover_random",
                         "large_subset_fine_tune"]:

            for fittest_weights in different_safety_weights_list:
                weights_crossover = copy.deepcopy(fittest_weights)

                model_offspring = keras_model_cnn(0, data)

                model_offspring.set_weights(weights_crossover)
                # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
                model_information_offspring = model_offspring.fit(x_train, y_train,
                                                                  epochs=total_training_epoch,
                                                                  batch_size=batch_size_sgd,
                                                                  verbose=2, validation_data=(x_test, y_test))

                list_performance = model_information_offspring.history["val_loss"]

                result_list.append(list_performance)

        keras.backend.clear_session()

    data_struc[str(work_id) + "_performance"] = result_list


if __name__ == "__main__":

    data = "cifar100"

    if data == "cifar10":
        x_train, x_test, y_train, y_test = load_cifar(False)
    elif data == "cifar100":
        x_train, x_test, y_train, y_test = load_cifar_100(False)
    elif data == "mnist":
        x_train, x_test, y_train, y_test = load_mnist(False)

    num_processes = 1

    start = timer()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    pair_list = [pair for pair in range(num_processes)]

    p = [multiprocessing.Process(target=crossover_offspring, args=(data, x_train, y_train, x_test, y_test,
                                                                   pair_list, i,
                                                                   return_dict)) for i in range(num_processes)]

    for proc in p:
        proc.start()
    for proc in p:
        proc.join()

    results = return_dict.values()

    pickle.dump(results, open("crossover_results.pickle", "wb"))

    end = timer()
    print(end - start)
