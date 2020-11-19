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
from utils import get_fittest_parent

from utils import compute_mask_convergence
from utils import identify_interesting_neurons
from utils import transplant_neurons
from utils import get_hidden_layers
from utils import compute_neurons_variance
from utils import match_random_filters
from utils import get_corr_cnn_filters
from utils import crossover_method
from utils import compute_q_values
from utils import mean_ensemble

from neural_models import CustomSaver
from neural_models import lr_scheduler
from neural_models import keras_model_cnn
from neural_models import keras_vgg

warnings.filterwarnings("ignore")


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc):
    # shuffle input data here

    num_pairs = len(pair_list)
    print("FOR PAIR NUMBER " + str(work_id + 1))

    np.random.seed(work_id + 1)
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    # program hyperparameters
    fittest_parent, weakest_parent = "parent_one", "parent_two"
    mix_full_networks = True
    total_training_epoch = 2
    batch_size_activation = 2048  # batch_size to compute the activation maps
    batch_size_sgd = 128
    cut_off = 0.2
    num_trainable_layer = 7

    if not mix_full_networks:
        x_train_s1, y_train_s1, x_test_s1, y_test_s1, x_train_s2, y_train_s2, x_test_s2, y_test_s2 = \
            partition_classes(x_train, x_test, y_train, y_test, cut_off)

    if mix_full_networks:
        crossover_types = ["frozen_aligned_targeted_crossover_low_corr", "frozen_aligned_targeted_crossover_random"]
        # crossover_types = ["aligned_targeted_crossover_low_corr", "aligned_targeted_crossover_random",
        #                   "mean_ensemble"]
    else:
        crossover_types = ["aligned_targeted_crossover_low_corr", "aligned_targeted_crossover_random",
                           "large_subset_fine_tune"]

    result_list = []
    epoch_list = np.arange(0, total_training_epoch, 1)

    model_full_dataset = keras_model_cnn(work_id, data)
    model_one = keras_model_cnn(work_id + num_pairs, data)
    model_two = keras_model_cnn(work_id + (2 * num_pairs), data)

    if os.path.isfile("best_epochs/best_epoch_parent_one_" + str(work_id)):
        print("loading existing weights...")
        best_epoch_parent_one = pickle.load(open("best_epochs/best_epoch_parent_one_" + str(work_id), "rb"))
        best_epoch_parent_two = pickle.load(open("best_epochs/best_epoch_parent_two_" + str(work_id), "rb"))
        loss_history_parent_full = pickle.load(open("best_epochs/history_parent_full_" + str(work_id), "rb"))

    else:

        if mix_full_networks:

            # train 2 parent networks on the full training data and use the fittest parent as benchmark

            save_callback = CustomSaver(epoch_list, "parent_one", work_id)
            model_information_parent_one = model_one.fit(x_train, y_train, epochs=total_training_epoch,
                                                         batch_size=batch_size_sgd,
                                                         verbose=2, validation_data=(x_test, y_test),
                                                         callbacks=[save_callback])

            save_callback = CustomSaver(epoch_list, "parent_two", work_id)
            model_information_parent_two = model_two.fit(x_train, y_train, epochs=total_training_epoch,
                                                         batch_size=batch_size_sgd,
                                                         verbose=2, validation_data=(x_test, y_test),
                                                         callbacks=[save_callback])

        else:
            # train a network on full data as benchmark and 2 other networks on subsets of the training data
            # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

            save_callback = CustomSaver(epoch_list, "parent_full", work_id)
            model_full_dataset = model_full_dataset.fit(x_train, y_train, epochs=total_training_epoch,
                                                        batch_size=batch_size_sgd,
                                                        verbose=2, validation_data=(x_test, y_test),
                                                        callbacks=[save_callback])

            save_callback = CustomSaver(epoch_list, "parent_one", work_id)
            model_information_parent_one = model_one.fit(x_train_s1, y_train_s1, epochs=total_training_epoch,
                                                         batch_size=batch_size_sgd, verbose=2,
                                                         validation_data=(x_test_s1, y_test_s1),
                                                         callbacks=[save_callback])

            save_callback = CustomSaver(epoch_list, "parent_two", work_id)
            model_information_parent_two = model_two.fit(x_train_s2, y_train_s2, epochs=total_training_epoch,
                                                         batch_size=batch_size_sgd, verbose=2,
                                                         validation_data=(x_test_s2, y_test_s2),
                                                         callbacks=[save_callback])

        # loss history of the fittest parent: the best parent must be called model_one
        if mix_full_networks:
            model_one, model_two, model_one_history, model_two_history, switch = get_fittest_parent(model_one,
                                                                                                    model_two,
                                                                                                    model_information_parent_one.history[
                                                                                                        "val_loss"],
                                                                                                    model_information_parent_two.history[
                                                                                                        "val_loss"],
                                                                                                    False)

            if switch:
                fittest_parent, weakest_parent = "parent_two", "parent_one"

            best_epoch_parent_one = np.argmin(model_one_history)
            best_epoch_parent_two = np.argmin(model_two_history)

            loss_history_parent_full = model_one_history

        else:
            loss_history_parent_full = model_full_dataset.history["val_loss"]
            best_epoch_parent_one = np.argmin(model_information_parent_one.history["val_loss"])
            best_epoch_parent_two = np.argmin(model_information_parent_two.history["val_loss"])

        pickle.dump(loss_history_parent_full, open("best_epochs/history_parent_full_" + str(work_id), "wb"))
        pickle.dump(best_epoch_parent_one, open("best_epochs/best_epoch_parent_one_" + str(work_id), "wb"))
        pickle.dump(best_epoch_parent_two, open("best_epochs/best_epoch_parent_two_" + str(work_id), "wb"))

    for crossover in crossover_types:

        print("crossover method: " + crossover)

        model_one = load_model(
            "parents_trained/model_" + fittest_parent + "_epoch_" + str(best_epoch_parent_one) + "_"
            + str(work_id) + ".hd5")
        weights_nn_one = model_one.get_weights()

        model_two = load_model(
            "parents_trained/model_" + weakest_parent + "_epoch_" + str(best_epoch_parent_two) + "_"
            + str(work_id) + ".hd5")
        weights_nn_two = model_two.get_weights()

        # compute hidden representation (e.g. activation maps for CNNs), on a batch of input

        list_hidden_representation_one = get_hidden_layers(model_one, x_test, batch_size_activation)
        list_hidden_representation_two = get_hidden_layers(model_two, x_test, batch_size_activation)
        # self-correlation matrix
        list_self_corr = get_corr_cnn_filters(list_hidden_representation_one, list_hidden_representation_one)
        # cross-correlation matrix
        list_cross_corr = get_corr_cnn_filters(list_hidden_representation_one, list_hidden_representation_two)

        if crossover in ["frozen_aligned_targeted_crossover_low_corr", "frozen_aligned_targeted_crossover_random"]:

            different_safety_weights_list = []

            for safety_level in ["safe_crossover", "naive_crossover"]:

                weights_parent = copy.deepcopy(weights_nn_one)
                hidden_representation_parent = copy.deepcopy(list_hidden_representation_one)

                depth = 0

                for layer in range(num_trainable_layer):

                    trainable_list = [True] * num_trainable_layer

                    if layer > 0:
                        trainable_list[:layer] = [False] * layer

                    print(trainable_list)

                    # train the network in a freeze train setting
                    model_offspring = keras_model_cnn(work_id + (3 * num_pairs), data, trainable_list)
                    if layer > 0:
                        model_offspring.set_weights(weights_offspring)
                    model_information_offspring = model_offspring.fit(x_train, y_train, batch_size=batch_size_sgd,
                                                                      epochs=int(
                                                                          total_training_epoch / num_trainable_layer),
                                                                      verbose=2, validation_data=(x_test, y_test))
                    weights_offspring = model_offspring.get_weights()

                    hidden_representation_offspring = get_hidden_layers(model_offspring, x_test, batch_size_activation)
                    list_corr_matrices = get_corr_cnn_filters(hidden_representation_offspring,
                                                              hidden_representation_parent)

                    list_cross_corr_copy = copy.deepcopy(list_corr_matrices)
                    list_ordered_indices_one, list_ordered_indices_two, weights_offspring, weights_parent = crossover_method(
                        weights_parent, weights_offspring, list_cross_corr_copy, safety_level)

                    model_parent = keras_model_cnn(0, data)
                    model_parent.set_weights(weights_parent)

                    model_offspring = keras_model_cnn(0, data)
                    model_offspring.set_weights(weights_offspring)

                    hidden_representation_offspring = get_hidden_layers(model_offspring, x_test, batch_size_activation)
                    hidden_representation_parent = get_hidden_layers(model_parent, x_test, batch_size_activation)
                    list_cross_corr = get_corr_cnn_filters(hidden_representation_offspring,
                                                           hidden_representation_parent)

                    var_list_best_parent = compute_neurons_variance(hidden_representation_offspring)
                    var_list_worst_parent = compute_neurons_variance(hidden_representation_parent)

                    # q_variance controls the variance threshold for deciding whether a filter is noise or not
                    q_variance = 0.1
                    mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, q_variance)
                    mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, q_variance)

                    # compute the q values for each layer (we use the same q values for naive alignment too)
                    if safety_level == "safe_crossover":
                        q_values_list = compute_q_values(list_ordered_indices_two, list_cross_corr)

                        if not mix_full_networks:
                            q_values_list = [cut_off] * len(list_cross_corr)

                    list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                        mask_convergence_best_parent,
                        mask_convergence_worst_parent,
                        list_cross_corr, list_self_corr, q_values_list)

                    if crossover == "aligned_targeted_crossover_random":
                        list_neurons_to_transplant, list_neurons_to_remove = match_random_filters(
                            mask_convergence_best_parent,
                            q_values_list)

                    # transplant layer by layer and order neurons after the transplant
                    weights_offspring = transplant_neurons(weights_offspring, weakest_parent,
                                                           list_neurons_to_transplant, list_neurons_to_remove, layer,
                                                           depth)

                    depth = (layer + 1) * 2

                    # modify the correlation matrix to reflect transplants and align the new layer in fittest weight
                    # with the layer in weakest weights (i.e. match the transplanted neurons with each other).

                    for index in range(len(list_neurons_to_transplant[layer])):
                        index_neurons_to_transplant = list_neurons_to_transplant[layer][index]
                        index_neurons_to_remove = list_neurons_to_remove[layer][index]

                        self_correlation_with_constraint = [-10000] * list_cross_corr[layer].shape[1]
                        self_correlation_with_constraint[index_neurons_to_transplant] = 0
                        list_cross_corr[layer][index_neurons_to_remove] = self_correlation_with_constraint

                    _, _, fittest_weights, weakest_weights = crossover_method(weights_offspring,
                                                                              weights_parent,
                                                                              list_cross_corr,
                                                                              safety_level)

                different_safety_weights_list.append(weights_offspring)

        if crossover in ["aligned_targeted_crossover_low_corr", "aligned_targeted_crossover_random"]:

            different_safety_weights_list = []

            for safety_level in ["safe_crossover", "naive_crossover"]:

                # first, functionally align the trained and initial networks
                list_cross_corr_copy = copy.deepcopy(list_cross_corr)
                fittest_weights, weakest_weights = copy.deepcopy(weights_nn_one), copy.deepcopy(weights_nn_two)

                list_ordered_indices_one, list_ordered_indices_two, fittest_weights, weakest_weights = crossover_method(
                    fittest_weights, weakest_weights, list_cross_corr_copy, safety_level)

                fittest_model = keras_model_cnn(0, data)
                fittest_model.set_weights(fittest_weights)

                weakest_model = keras_model_cnn(0, data)
                weakest_model.set_weights(weakest_weights)

                # get the reshuffled correlation matrix for the aligned networks
                hidden_representation_fittest = get_hidden_layers(fittest_model, x_test, batch_size_activation)
                hidden_representation_weakest = get_hidden_layers(weakest_model, x_test, batch_size_activation)
                list_cross_corr = get_corr_cnn_filters(hidden_representation_fittest, hidden_representation_weakest)

                var_list_best_parent = compute_neurons_variance(hidden_representation_fittest)
                var_list_worst_parent = compute_neurons_variance(hidden_representation_weakest)

                # q_variance controls the variance threshold for deciding whether a filter is noise or not
                q_variance = 0.1
                mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, q_variance)
                mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, q_variance)

                # compute the q values for each layer (we use the same q values for naive alignment too)
                if safety_level == "safe_crossover":
                    q_values_list = compute_q_values(list_ordered_indices_two, list_cross_corr)

                    if not mix_full_networks:
                        q_values_list = [cut_off] * len(list_cross_corr)

                list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                    mask_convergence_best_parent,
                    mask_convergence_worst_parent,
                    list_cross_corr, list_self_corr, q_values_list)

                if crossover == "aligned_targeted_crossover_random":
                    list_neurons_to_transplant, list_neurons_to_remove = match_random_filters(
                        mask_convergence_best_parent,
                        q_values_list)

                depth = 0

                for layer in range(len(list_neurons_to_transplant)):
                    # transplant layer by layer and order neurons after the transplant
                    fittest_weights = transplant_neurons(fittest_weights, weakest_weights,
                                                         list_neurons_to_transplant, list_neurons_to_remove, layer,
                                                         depth)

                    depth = (layer + 1) * 2

                    # modify the correlation matrix to reflect transplants and align the new layer in fittest weight
                    # with the layer in weakest weights (i.e. match the transplanted neurons with each other).

                    for index in range(len(list_neurons_to_transplant[layer])):
                        index_neurons_to_transplant = list_neurons_to_transplant[layer][index]
                        index_neurons_to_remove = list_neurons_to_remove[layer][index]

                        self_correlation_with_constraint = [-10000] * list_cross_corr[layer].shape[1]
                        self_correlation_with_constraint[index_neurons_to_transplant] = 0
                        list_cross_corr[layer][index_neurons_to_remove] = self_correlation_with_constraint

                    _, _, fittest_weights, weakest_weights = crossover_method(fittest_weights,
                                                                              weakest_weights,
                                                                              list_cross_corr,
                                                                              safety_level)

                different_safety_weights_list.append(fittest_weights)

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

    result_list.append(loss_history_parent_full)

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
