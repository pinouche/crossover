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

from utils import load_model_long_string
from utils import compute_mask_convergence
from utils import identify_interesting_neurons
from utils import transplant_neurons
from utils import apply_mask_and_add_noise
from utils import get_gradients_hidden_layers
from utils import get_hidden_layers
from utils import get_gradient_weights
from utils import get_magnitude_weights
from utils import get_movement_weights
from utils import check_convergence
from utils import get_corr
from utils import crossover_method
from utils import arithmetic_crossover
from utils import add_noise_to_fittest
from utils import corr_neurons
from utils import scale_fittest_parent

from feed_forward import CustomSaver
from feed_forward import model_keras

warnings.filterwarnings("ignore")


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc, parallel="process"):
    # shuffle input data here
    # np.random.seed(work_id)
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    num_pairs = len(pair_list)

    print("FOR PAIR NUMBER " + str(work_id + 1))

    # crossover_types = ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover",
    # "naive_crossover", # "noise_low_corr", "noise_high_corr", "safe_mutation_gradient", "unsafe_mutation_gradient",
    # "safe_mutation_magnitude", "unsafe_mutation_magnitude", "safe_mutation_movement", "unsafe_mutation_movement",
    # "safe_mutation_convergence_neurons", "unsafe_mutation_convergence_neurons", "noise_0.5", "noise_0.1",
    # "aligned_targeted_crossover", "scale_fittest_parent"]
    crossover_types = ["aligned_targeted_crossover"]

    vector_representation = "activation"  # "gradient" or "activation"
    result_list = [[] for _ in range(len(crossover_types)+1)]
    quantile = 0.5
    total_training_epoch = 60
    epoch_list = np.arange(0, total_training_epoch, 1)

    model_one = model_keras(work_id, data)
    model_two = model_keras(work_id + num_pairs, data)
    model_one.save("parents_initial/parent_one_initial_" + str(work_id) + ".hd5")
    model_two.save("parents_initial/parent_two_initial_" + str(work_id) + ".hd5")

    print("one")
    save_callback = CustomSaver(epoch_list, "parent_one", work_id)
    model_information_parent_one = model_one.fit(x_train, y_train, epochs=total_training_epoch, batch_size=512,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])

    save_callback = CustomSaver(epoch_list, "parent_two", work_id)
    model_information_parent_two = model_two.fit(x_train, y_train, epochs=total_training_epoch, batch_size=512,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])
    print("three")

    counter = 0
    for crossover in crossover_types:

        # get the parents' weights at the best epoch
        best_epoch_parent_one = np.argmin(model_information_parent_one.history["val_loss"])
        parent_one = load_model("parents_trained/model_parent_one_epoch_" + str(best_epoch_parent_one) + "_" + str(work_id) + ".hd5")
        weights_nn_one = parent_one.get_weights()

        best_epoch_parent_two = np.argmin(model_information_parent_two.history["val_loss"])
        parent_two = load_model("parents_trained/model_parent_two_epoch_" + str(best_epoch_parent_two) + "_" + str(work_id) + ".hd5")
        weights_nn_two = parent_two.get_weights()

        fittest_weights, fittest_model, best_initial_id = weights_nn_one, parent_one, "parent_one"
        weakest_weights, weakest_model, weakest_initial_id = weights_nn_two, parent_two, "parent_two"
        loss_best_parent = model_information_parent_one.history["val_loss"]
        if np.min(loss_best_parent) > np.min(model_information_parent_two.history["val_loss"]):
            loss_best_parent = model_information_parent_two.history["val_loss"]
            fittest_weights, fittest_model, best_initial_id = weights_nn_two, parent_two, "parent_two"
            weakest_weights, weakest_model, weakest_initial_id = weights_nn_one, parent_one, "parent_one"

        print("crossover method: " + crossover)
        list_ordered_weights_one, list_ordered_weights_two = weights_nn_one, weights_nn_two

        if crossover in ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover",
                         "naive_crossover", "noise_low_corr", "noise_high_corr", "aligned_targeted_crossover"]:

            if vector_representation == "activation":
                list_hidden_representation_fittest = get_hidden_layers(fittest_model, x_test)  # activation vector network one
                list_hidden_representation_weakest = get_hidden_layers(weakest_model, x_test)  # activation vector network two
            elif vector_representation == "gradient":
                list_hidden_representation_fittest = get_gradients_hidden_layers(fittest_model, x_test, y_test)  # gradient vector
                list_hidden_representation_weakest = get_gradients_hidden_layers(weakest_model, x_test, y_test)  # gradient vector

            if crossover not in ["noise_low_corr", "noise_high_corr"]:
                list_corr_matrices = get_corr(list_hidden_representation_fittest, list_hidden_representation_weakest)
            else:
                list_corr_matrices = get_corr(list_hidden_representation_fittest, list_hidden_representation_fittest)

        # based on the two parents
        if crossover in ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover", "naive_crossover"]:
            list_ordered_weights_one, list_ordered_weights_two, parents_similarity = crossover_method(
                list_ordered_weights_one, list_ordered_weights_two, list_corr_matrices, crossover)

        # based only on fittest parent
        if crossover in ["noise_0.5", "noise_0.1"]:
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id)

        # based only on fittest parent
        elif crossover in ["safe_mutation_magnitude", "unsafe_mutation_magnitude"]:
            sensitivity_vector = get_magnitude_weights(fittest_weights)
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id, sensitivity_vector,
                                                     crossover)

        # based only on fittest parent
        elif crossover in ["safe_mutation_gradient", "unsafe_mutation_gradient"]:
            sensitivity_vector = get_gradient_weights(fittest_model, x_test)
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id, sensitivity_vector, crossover)

        # based only on fittest parent
        elif crossover in ["safe_mutation_movement", "unsafe_mutation_movement"]:
            sensitivity_vector = get_movement_weights(fittest_weights, best_initial_id, work_id)
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id, sensitivity_vector,
                                                     crossover)

        elif crossover in ["safe_mutation_convergence_neurons", "unsafe_mutation_convergence_neurons",
                           "aligned_targeted_crossover"]:

            activation_epochs_best_parent, activation_epochs_worst_parent = [], []
            for index in np.arange(0, total_training_epoch, 5):
                best_model = load_model_long_string(best_initial_id, work_id, index)
                worst_model = load_model_long_string(weakest_initial_id, work_id, index)
                hidden_representation_best_parent = get_hidden_layers(best_model, x_test)
                hidden_representation_worst_parent = get_hidden_layers(worst_model, x_test)
                activation_epochs_best_parent.append(hidden_representation_best_parent)
                activation_epochs_worst_parent.append(hidden_representation_worst_parent)
            activation_epochs_best_parent = np.array(activation_epochs_best_parent)
            activation_epochs_worst_parent = np.array(activation_epochs_worst_parent)

            var_list_best_parent, corr_list_best_parent = check_convergence(activation_epochs_best_parent)
            var_list_worst_parent, corr_list_worst_parent = check_convergence(activation_epochs_worst_parent)
            mask_convergence_best_parent = compute_mask_convergence(var_list_best_parent, corr_list_best_parent)
            mask_convergence_worst_parent = compute_mask_convergence(var_list_worst_parent, corr_list_worst_parent)

            # this function identifies neurons from the weaker parent
            list_neurons_to_transplant, list_neurons_to_remove = identify_interesting_neurons(
                                                                      mask_convergence_best_parent,
                                                                      mask_convergence_worst_parent,
                                                                      list_corr_matrices)
            print(list_neurons_to_transplant, list_neurons_to_remove)

            # first, functionally align the networks
            list_corr_matrices_copy = list_corr_matrices.copy()
            fittest_weights, weakest_weights, _ = crossover_method(fittest_weights, weakest_weights,
                                                                   list_corr_matrices_copy, "safe_crossover")

            count = 0
            depth = 0

            for layer in range(len(list_neurons_to_transplant)):
                print(layer)
                # transplant layer by layer and order neurons after the transplant
                fittest_weights = transplant_neurons(fittest_weights, weakest_weights, list_neurons_to_transplant,
                                                     list_neurons_to_remove, layer, depth)

                count += 1
                depth = count * 2

                # modify the correlation matrix to reflect transplants and align the new layer in fittest weight with
                # the layer in weakest weights (i.e. match the transplanted neurons with each other).

                for index in range(len(list_neurons_to_transplant[layer])):
                    index_neurons_to_transplant = list_neurons_to_transplant[layer][index]
                    index_neurons_to_remove = list_neurons_to_remove[layer][index]

                    self_correlation_with_constraint = [-10000] * list_corr_matrices[layer].shape[1]
                    self_correlation_with_constraint[index_neurons_to_transplant] = 0
                    list_corr_matrices[layer][index_neurons_to_remove] = self_correlation_with_constraint

                list_corr_matrices_copy = list_corr_matrices.copy()
                fittest_weights, weakest_weights, _ = crossover_method(fittest_weights, weakest_weights,
                                                                       list_corr_matrices_copy,
                                                                       "safe_crossover")
            weights_crossover = fittest_weights

            # this is for safe and unsafe mutation convergence neurons
            #  if "safe" in crossover:
            #    mask_convergence = [~mask for mask in mask_convergence]
            #  weights_crossover = apply_mask_and_add_noise(fittest_weights, mask_convergence, work_id)

        # based only on fittest parent
        elif crossover in ["noise_low_corr", "noise_high_corr"]:
            weights_crossover = corr_neurons(fittest_weights, list_corr_matrices, work_id, crossover, quantile)

        elif crossover in ["scale_fittest_parent"]:
            weights_crossover = scale_fittest_parent(fittest_weights)

        else:
            weights_crossover = arithmetic_crossover(list_ordered_weights_one, list_ordered_weights_two)

        model_offspring = model_keras(0, data, 0.0002, weights_crossover[0].shape[1])

        model_offspring.set_weights(weights_crossover)
        model_information_offspring = model_offspring.fit(x_train, y_train,
                                                          epochs=total_training_epoch,
                                                          batch_size=512,
                                                          verbose=True, validation_data=(x_test, y_test))

        if counter == 0:
            result_list[counter].append(loss_best_parent)
        result_list[counter+1].append(model_information_offspring.history["val_loss"])

        keras.backend.clear_session()
        counter += 1

    if parallel == "process":
        data_struc[str(work_id) + "_performance"] = result_list

    print("ten")


if __name__ == "__main__":

    data = "cifar10"

    if data == "cifar10":
        x_train, x_test, y_train, y_test = load_cifar()
    elif data == "cifar100":
        x_train, x_test, y_train, y_test = load_cifar_100()
    elif data == "mnist":
        x_train, x_test, y_train, y_test = load_mnist()

    parallel_method = "process"

    if parallel_method == "process":
        num_processes = 1

        start = timer()

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        pair_list = [pair for pair in range(num_processes)]

        p = [multiprocessing.Process(target=crossover_offspring, args=(data, x_train, y_train, x_test, y_test,
                                                                       pair_list, i, return_dict,
                                                                       parallel_method)) for i in range(num_processes)]

        for proc in p:
            proc.start()
        for proc in p:
            proc.join()

        results = return_dict.values()

        pickle.dump(results, open("crossover_results.pickle", "wb"))

        end = timer()
        print(end - start)
