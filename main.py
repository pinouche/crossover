import numpy as np
from timeit import default_timer as timer
import multiprocessing
import warnings
import pickle

from keras.models import load_model
import keras

from utils import load_cifar
from utils import load_mnist
from utils import load_cifar_100
from utils import partition_classes

from utils import compute_mask_convergence
from utils import identify_interesting_neurons
from utils import transplant_neurons
from utils import get_gradients_hidden_layers
from utils import get_hidden_layers
from utils import compute_neurons_variance
from utils import get_corr_cnn_filters
from utils import crossover_method
from utils import arithmetic_crossover

from neural_models import CustomSaver
from neural_models import keras_model_cnn
from neural_models import linear_classifier_keras

warnings.filterwarnings("ignore")


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc):
    # shuffle input data here
    np.random.seed(work_id + 1)
    shuffle_list = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_list)
    x_train = x_train[shuffle_list]
    y_train = y_train[shuffle_list]

    x_train_s1, y_train_s1, x_test_s1, y_test_s1, x_train_s2, y_train_s2, x_test_s2, y_test_s2 = \
        partition_classes(x_train, x_test, y_train, y_test)

    num_pairs = len(pair_list)

    print("FOR PAIR NUMBER " + str(work_id + 1))

    # crossover_types = ["aligned_targeted_crossover_high_corr", "aligned_targeted_crossover_low_corr",
    #                   "aligned_targeted_crossover_variance", "fine_tune_fittest"]

    crossover_types = ["aligned_targeted_crossover_low_corr", "fine_tune_fittest"]

    vector_representation = "activation"  # "gradient" or "activation"
    result_list = []
    total_training_epoch = 2
    epoch_list = np.arange(0, total_training_epoch, 1)

    model_full_dataset = keras_model_cnn(work_id, data)
    model_one = keras_model_cnn(work_id + num_pairs, data)
    model_two = keras_model_cnn(work_id + (2 * num_pairs), data)
    model_one.save("parents_initial/parent_one_initial_" + str(work_id) + ".hd5")
    model_two.save("parents_initial/parent_two_initial_" + str(work_id) + ".hd5")
    weights_initial_one = model_one.get_weights()
    weights_initial_two = model_two.get_weights()

    save_callback = CustomSaver(epoch_list, "parent_full", work_id)
    parent_full_dataset = model_full_dataset.fit(x_train, y_train, epochs=total_training_epoch, batch_size=512,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])

    save_callback = CustomSaver(epoch_list, "parent_one", work_id)
    model_information_parent_one = model_one.fit(x_train_s1, y_train_s1, epochs=total_training_epoch, batch_size=512,
                                                 verbose=False,
                                                 validation_data=(x_test_s1, y_test_s1), callbacks=[save_callback])

    save_callback = CustomSaver(epoch_list, "parent_two", work_id)
    model_information_parent_two = model_two.fit(x_train_s2, y_train_s2, epochs=total_training_epoch, batch_size=512,
                                                 verbose=False,
                                                 validation_data=(x_test_s2, y_test_s2), callbacks=[save_callback])

    # get the parents' weights at the best epoch and retrieve the initial weights
    best_epoch_parent_one = np.argmin(model_information_parent_one.history["val_loss"])
    best_epoch_parent_two = np.argmin(model_information_parent_two.history["val_loss"])

    for crossover in crossover_types:

        parent_one = load_model(
            "parents_trained/model_parent_one_epoch_" + str(best_epoch_parent_one) + "_" + str(work_id) + ".hd5")
        weights_nn_one = parent_one.get_weights()

        parent_two = load_model(
            "parents_trained/model_parent_two_epoch_" + str(best_epoch_parent_two) + "_" + str(work_id) + ".hd5")
        weights_nn_two = parent_two.get_weights()

        print("crossover method: " + crossover)

        if crossover in ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover",
                         "aligned_targeted_crossover_high_corr", "aligned_targeted_crossover_low_corr",
                         "aligned_targeted_crossover_variance", "feature_extraction_variance",
                         "feature_extraction_random", "feature_extraction_low_corr", "feature_extraction_high_corr"]:

            if vector_representation == "activation":
                list_hidden_representation_one = get_hidden_layers(model_one, x_test, 2058)
                list_hidden_representation_two = get_hidden_layers(model_two, x_test, 2058)
            else:
                list_hidden_representation_one = get_gradients_hidden_layers(model_one, x_test, y_test)
                list_hidden_representation_two = get_gradients_hidden_layers(model_two, x_test, y_test)

            list_corr_matrices = get_corr_cnn_filters(list_hidden_representation_one, list_hidden_representation_two)

        # based on the two parents
        if crossover in ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover",
                         "naive_crossover"]:
            fittest_weights, weakest_weights = crossover_method(weights_nn_one, weights_nn_two,
                                                                list_corr_matrices, crossover)

        elif crossover in ["feature_extraction_variance", "feature_extraction_random", "feature_extraction_low_corr",
                           "feature_extraction_high_corr"]:

            extracted_x_train_fittest = get_hidden_layers(model_one, x_train_s1)[-1]
            extracted_x_test_fittest = list_hidden_representation_one[-1]

            extracted_x_train_weakest = get_hidden_layers(model_two, x_train_s2)[-1]
            extracted_x_test_weakest = list_hidden_representation_one[-1]

            var_list_best_parent = compute_neurons_variance(extracted_x_test_fittest)[0]
            var_list_worst_parent = compute_neurons_variance(extracted_x_test_weakest)[0]

            if crossover == "feature_extraction_variance":
                q_fittest = 0.25
                q_worst = 1 - q_fittest
                mask_convergence_fittest = compute_mask_convergence(var_list_best_parent, q_fittest)[0]
                mask_convergence_weakest = compute_mask_convergence(var_list_worst_parent, q_worst)[0]

                extracted_x_train_fittest[:, mask_convergence_fittest] = extracted_x_train_weakest[:,
                                                                         ~mask_convergence_weakest]
                extracted_x_test_fittest[:, mask_convergence_fittest] = extracted_x_test_weakest[:,
                                                                        ~mask_convergence_weakest]

            elif crossover in ["feature_extraction_high", "feature_extraction_low"]:

                q = 0.25
                mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, q)
                mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, q)

                # this function identifies neurons from the weaker parent
                corr_wanted = "low"
                if crossover == "feature_extraction_high":
                    corr_wanted = "high"

                list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                    mask_convergence_best_parent,
                    mask_convergence_worst_parent,
                    list_corr_matrices, corr_wanted)

                extracted_x_train_fittest[:, list_neurons_to_remove] = extracted_x_train_weakest[:,
                                                                       list_neurons_to_transplant]
                extracted_x_test_fittest[:, list_neurons_to_remove] = extracted_x_test_weakest[:,
                                                                      list_neurons_to_transplant]

            elif crossover == "feature_extraction_random":
                q = 0.25
                num_neurons = extracted_x_train_fittest.shape[1]
                bool_vector_fittest = np.array(
                    [0] * int(num_neurons * (1 - q)) + [1] * int(num_neurons * q))
                np.random.shuffle(bool_vector_fittest)
                bool_vector_fittest = bool_vector_fittest.astype(bool)

                bool_vector_weakest = np.array(
                    [0] * int(num_neurons * (1 - q)) + [1] * int(num_neurons * q))
                np.random.shuffle(bool_vector_weakest)
                bool_vector_weakest = bool_vector_weakest.astype(bool)

                extracted_x_train_fittest[:, bool_vector_fittest] = extracted_x_train_weakest[:, bool_vector_weakest]
                extracted_x_test_fittest[:, bool_vector_fittest] = extracted_x_test_weakest[:, bool_vector_weakest]

            model_offspring = linear_classifier_keras(0, extracted_x_train_fittest.shape[1], data)

            model_information_offspring = model_offspring.fit(extracted_x_train_fittest, y_train,
                                                              epochs=total_training_epoch,
                                                              batch_size=512,
                                                              verbose=True,
                                                              validation_data=(extracted_x_test_fittest, y_test))

            list_performance = model_information_offspring.history["val_loss"]

            result_list.append(list_performance)

        elif crossover in ["aligned_targeted_crossover_high_corr", "aligned_targeted_crossover_low_corr",
                           "aligned_targeted_crossover_variance"]:

            # first, functionally align the trained and initial networks
            list_corr_matrices_copy = list_corr_matrices.copy()
            fittest_weights, weakest_weights = crossover_method(weights_nn_one, weights_nn_two,
                                                                list_corr_matrices_copy, "safe_crossover")

            fittest_model = keras_model_cnn(0, data)
            fittest_model.set_weights(fittest_weights)

            weakest_model = keras_model_cnn(0, data)
            weakest_model.set_weights(weakest_weights)

            hidden_representation_fittest = get_hidden_layers(fittest_model, x_test, 2058)
            hidden_representation_weakest = get_hidden_layers(weakest_model, x_test, 2058)

            # get the reshuffled correlation matrix for the aligned networks
            list_corr_matrices = get_corr_cnn_filters(hidden_representation_fittest, hidden_representation_weakest)

            var_list_best_parent = compute_neurons_variance(hidden_representation_fittest)
            var_list_worst_parent = compute_neurons_variance(hidden_representation_weakest)

            q_fittest = 0.2
            q_worst = q_fittest
            if crossover == "aligned_targeted_crossover_variance":
                q_worst = 1 - q_fittest

            mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, q_fittest)
            mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, q_worst)

            # this function identifies neurons from the weaker parent
            corr_wanted = "low"
            if crossover == "aligned_targeted_crossover_high_corr":
                corr_wanted = "high"

            list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                mask_convergence_best_parent,
                mask_convergence_worst_parent,
                list_corr_matrices, corr_wanted)

            depth = 0

            for layer in range(len(list_neurons_to_transplant)):
                # transplant layer by layer and order neurons after the transplant
                fittest_weights = transplant_neurons(fittest_weights, weakest_weights,
                                                     list_neurons_to_transplant, list_neurons_to_remove, layer, depth)

                depth = (layer + 1) * 2

                # modify the correlation matrix to reflect transplants and align the new layer in fittest weight with
                # the layer in weakest weights (i.e. match the transplanted neurons with each other).

                for index in range(len(list_neurons_to_transplant[layer])):
                    index_neurons_to_transplant = list_neurons_to_transplant[layer][index]
                    index_neurons_to_remove = list_neurons_to_remove[layer][index]

                    self_correlation_with_constraint = [-10000] * list_corr_matrices[layer].shape[1]
                    self_correlation_with_constraint[index_neurons_to_transplant] = 0
                    list_corr_matrices[layer][index_neurons_to_remove] = self_correlation_with_constraint

                list_corr_matrices_copy = list_corr_matrices.copy()
                fittest_weights, weakest_weights = crossover_method(fittest_weights,
                                                                    weakest_weights,
                                                                    list_corr_matrices_copy,
                                                                    "safe_crossover")

        elif crossover in ["aligned_targeted_freeze_training_crossover"]:

            list_performance = []
            number_of_hidden_layers = len(model_one.layers) - 2
            fittest_weights, weakest_weights = weights_initial_one, weights_initial_two
            number_epochs_interval = int(total_training_epoch / (number_of_hidden_layers + 1))
            trainable_list = [True] * (number_of_hidden_layers + 1)
            depth = 0

            for layer in range(number_of_hidden_layers):

                if layer > 0:
                    trainable_list[:layer] = [False] * len(trainable_list[:layer])

                model_fittest = keras_model_cnn(0, data, trainable_list)
                model_weakest = keras_model_cnn(0, data, trainable_list)

                model_fittest.set_weights(fittest_weights)
                model_weakest.set_weights(weakest_weights)

                model_information_offspring = model_fittest.fit(x_train, y_train,
                                                                epochs=number_epochs_interval,
                                                                batch_size=512,
                                                                verbose=True, validation_data=(x_test, y_test))

                list_performance.append(model_information_offspring.history["val_loss"])

                model_weakest.fit(x_train, y_train,
                                  epochs=number_epochs_interval,
                                  batch_size=512,
                                  verbose=False, validation_data=(x_test, y_test))

                hidden_representation_fittest = get_hidden_layers(model_fittest, x_test)
                hidden_representation_weakest = get_hidden_layers(model_weakest, x_test)
                list_corr_matrices = get_corr_cnn_filters(hidden_representation_fittest, hidden_representation_weakest)

                fittest_weights = model_fittest.get_weights()
                weakest_weights = model_weakest.get_weights()

                # functionally align the networks
                list_corr_matrices_copy = list_corr_matrices.copy()
                fittest_weights, weakest_weights = crossover_method(fittest_weights, weakest_weights,
                                                                    list_corr_matrices_copy, "safe_crossover")

                # get the reshuffled correlation matrix for the aligned networks
                hidden_representation_fittest = get_hidden_layers(fittest_model, x_test)
                hidden_representation_weakest = get_hidden_layers(weakest_model, x_test)
                list_corr_matrices = get_corr_cnn_filters(hidden_representation_fittest, hidden_representation_weakest)

                var_list_best_parent = compute_neurons_variance(hidden_representation_fittest)
                var_list_worst_parent = compute_neurons_variance(hidden_representation_weakest)

                q = 0.2
                mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, q)
                mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, q - 1)

                list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                    mask_convergence_best_parent,
                    mask_convergence_worst_parent,
                    list_corr_matrices)

                fittest_weights = transplant_neurons(fittest_weights, weakest_weights,
                                                     list_neurons_to_transplant, list_neurons_to_remove, layer, depth)

                depth = (layer + 1) * 2

                for index in range(len(list_neurons_to_transplant[layer])):
                    index_neurons_to_transplant = list_neurons_to_transplant[layer][index]
                    index_neurons_to_remove = list_neurons_to_remove[layer][index]

                    self_correlation_with_constraint = [-10000] * list_corr_matrices[layer].shape[1]
                    self_correlation_with_constraint[index_neurons_to_transplant] = 0
                    list_corr_matrices[layer][index_neurons_to_remove] = self_correlation_with_constraint

                list_corr_matrices_copy = list_corr_matrices.copy()
                fittest_weights, weakest_weights = crossover_method(fittest_weights,
                                                                    weakest_weights,
                                                                    list_corr_matrices_copy,
                                                                    "safe_crossover")
                if layer < 2:
                    for index in range(depth, len(fittest_weights)):
                        # weight matrix
                        if index % 2 == 0:
                            fittest_weights[index] = np.random.normal(loc=0.0,
                                                                      scale=np.sqrt(
                                                                          2 / (fittest_weights[index].shape[0] +
                                                                               fittest_weights[index].shape[1])),
                                                                      size=fittest_weights[index].shape)
                        else:
                            fittest_weights[index] = np.zeros(fittest_weights[index].shape)

            trainable_list = [False] * (number_of_hidden_layers + 1)
            trainable_list[-1] = True
            model_fittest = keras_model_cnn(0, data, trainable_list)

            # reset the weights of the last linear layer (Glorot normal)
            fittest_weights[-1] = np.random.normal(loc=0.0, scale=np.sqrt(2 / (fittest_weights[-1].shape[0] +
                                                                               fittest_weights[-1].shape[1])),
                                                   size=fittest_weights[-1].shape)

            model_fittest.set_weights(fittest_weights)

            model_information_offspring = model_fittest.fit(x_train, y_train,
                                                            epochs=number_epochs_interval,
                                                            batch_size=512,
                                                            verbose=True, validation_data=(x_test, y_test))

            list_performance.append(model_information_offspring.history["val_loss"])
            list_performance = [val for sub_list in list_performance for val in sub_list]
            result_list.append(list_performance)

        elif crossover in ["fine_tune_fittest"]:
            fittest_weights = weights_nn_one.copy()

        else:
            fittest_weights = arithmetic_crossover(fittest_weights, weakest_weights)

        if crossover in ["aligned_targeted_crossover_high_corr", "aligned_targeted_crossover_low_corr",
                         "aligned_targeted_crossover_variance", "fine_tune_fittest"]:

            # "fine_tune" or "fixed"
            for transfer_style in ["fine_tune"]:
                print(transfer_style)

                weights_crossover = fittest_weights.copy()

                # fine tuning vs transfer learning with fixed lower layers
                num_trainable_layers = int((len(model_one.trainable_weights)/2)+0.5)
                trainable_list = [True] * num_trainable_layers
                if transfer_style == "fixed":
                    trainable_list[:-1] = [False] * (num_trainable_layers-1)
                    trainable_list[-1] = True

                model_offspring = keras_model_cnn(0, data, trainable_list)

                # reset the weights of the last linear layer (Glorot normal)
                weights_crossover[-1] = np.random.normal(loc=0.0, scale=np.sqrt(2 / (weights_crossover[-1].shape[0] +
                                                                                     weights_crossover[-1].shape[1])),
                                                         size=weights_crossover[-1].shape)

                model_offspring.set_weights(weights_crossover)
                model_information_offspring = model_offspring.fit(x_train, y_train,
                                                                  epochs=total_training_epoch,
                                                                  batch_size=512,
                                                                  verbose=True, validation_data=(x_test, y_test))

                list_performance = model_information_offspring.history["val_loss"]

                result_list.append(list_performance)

        keras.backend.clear_session()

    result_list.append(parent_full_dataset.history["val_loss"])

    data_struc[str(work_id) + "_performance"] = result_list

    print("ten")


if __name__ == "__main__":

    data = "cifar10"

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
