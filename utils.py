import numpy as np
from scipy.optimize import linear_sum_assignment
import keras
import copy
import random
import operator


def get_hidden_layers(model, data_x, batch_size):
    data_x = data_x[:batch_size]

    def keras_function_layer(model_layer, data):
        hidden_func = keras.backend.function(model.layers[0].input, model_layer.output)
        result = hidden_func([data])

        return result

    hidden_layers_list = []
    for index in range(len(model.layers)):
        if isinstance(model.layers[index], keras.layers.convolutional.Conv2D) or isinstance(model.layers[index],
                                                                                            keras.layers.Dense):
            hidden_layer = keras_function_layer(model.layers[index], data_x)
            hidden_layers_list.append(hidden_layer)

    return hidden_layers_list


def compute_neurons_variance(hidden_layers_list):
    list_variance_filters = []
    
    for layer_id in range(len(hidden_layers_list) - 2):

        batch_size = hidden_layers_list[layer_id].shape[0]
        size_activation_map = hidden_layers_list[layer_id].shape[1]

        # draw a random value from each of the CNN filters
        i_dim = np.random.choice(range(0, size_activation_map), batch_size)
        j_dim = np.random.choice(range(0, size_activation_map), batch_size)

        layer_one = []
        for index in range(batch_size):
            layer_one.append(hidden_layers_list[layer_id][index][i_dim[index], j_dim[index], :])

        variance = np.var(np.array(layer_one), axis=0)
        list_variance_filters.append(variance)

    return list_variance_filters


def match_random_filters(q_value_list, list_cross_corr):
    filters_to_remove = []
    filters_to_transplant = []

    for index in range(len(q_value_list)):
        num_filters = int(list_cross_corr[index].shape[0]*q_value_list[index])
        num_filters_to_change = int(num_filters * q_value_list[index])
        indices_to_remove = random.sample(range(num_filters), num_filters_to_change)
        indices_to_transplant = random.sample(range(num_filters), num_filters_to_change)

        filters_to_remove.append(indices_to_remove)
        filters_to_transplant.append(indices_to_transplant)

    return filters_to_transplant, filters_to_remove


def get_corr_cnn_filters(hidden_representation_list_one, hidden_representation_list_two):
    list_corr_matrices = []

    for layer_id in range(len(hidden_representation_list_one) - 2):

        batch_size = hidden_representation_list_one[layer_id].shape[0]
        size_activation_map = hidden_representation_list_one[layer_id].shape[1]
        num_filters = hidden_representation_list_one[layer_id].shape[-1]

        # draw a random value from each of the CNN filters
        i_dim = np.random.choice(range(0, size_activation_map), batch_size)
        j_dim = np.random.choice(range(0, size_activation_map), batch_size)

        layer_one = []
        layer_two = []
        for index in range(batch_size):
            layer_one.append(hidden_representation_list_one[layer_id][index][i_dim[index], j_dim[index], :])
            layer_two.append(hidden_representation_list_two[layer_id][index][i_dim[index], j_dim[index], :])

        layer_one = np.array(layer_one)
        layer_two = np.array(layer_two)

        cross_corr_matrix = np.corrcoef(layer_one, layer_two, rowvar=False)[num_filters:, :num_filters]

        cross_corr_matrix[np.isnan(cross_corr_matrix)] = 0
        list_corr_matrices.append(cross_corr_matrix)

    return list_corr_matrices


# cross correlation function for both bipartite matching (hungarian method)
def bipartite_matching(corr_matrix_nn, crossover="safe_crossover"):
    corr_matrix_nn_tmp = copy.deepcopy(corr_matrix_nn)
    if crossover == "unsafe_crossover":
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn_tmp)
    elif crossover == "safe_crossover":
        corr_matrix_nn_tmp *= -1  # default of linear_sum_assignement is to minimize cost, we want to max "cost"
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn_tmp)
    elif crossover == "orthogonal_crossover":
        corr_matrix_nn_tmp = np.abs(corr_matrix_nn_tmp)
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn_tmp)
    elif crossover == "normed_crossover":
        corr_matrix_nn_tmp = np.abs(corr_matrix_nn_tmp)
        corr_matrix_nn_tmp *= -1
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn_tmp)
    elif crossover == "naive_crossover":
        list_neurons_x, list_neurons_y = list(range(corr_matrix_nn_tmp.shape[0])), list(range(corr_matrix_nn_tmp.shape[0]))
    else:
        raise ValueError('the crossover method is not defined')

    return list_neurons_x, list_neurons_y


# Algorithm 2
def permute_cnn(weights_list_copy, list_permutation):
    depth = 0
    for layer in range(len(list_permutation)):
        for index in range(7):
            if index == 0:
                # order filters
                weights_list_copy[index + depth] = weights_list_copy[index + depth][:, :, :, list_permutation[layer]]
            elif index in [1, 2, 3, 4, 5]:
                # order the biases and the batch norm parameters
                weights_list_copy[index + depth] = weights_list_copy[index + depth][list_permutation[layer]]
            elif index == 6:
                if (index + depth) != (len(weights_list_copy) - 3):
                    # order channels
                    weights_list_copy[index + depth] = weights_list_copy[index + depth][:, :, list_permutation[layer],
                                                       :]
                else:  # this is for the flattened fully connected layer

                    num_filters = len(list_permutation[layer])
                    weights_tmp = copy.deepcopy(weights_list_copy[index + depth])
                    activation_map_size = int(weights_tmp.shape[0] / num_filters)

                    for i in range(num_filters):
                        filter_id = list_permutation[layer][i]
                        weights_list_copy[index + depth][[num_filters * j + i for j in range(activation_map_size)]] = \
                            weights_tmp[[num_filters * j + filter_id for j in range(activation_map_size)]]

        depth = (layer + 1) * 6

    return weights_list_copy


def transplant_neurons(fittest_weights, weakest_weights, indices_transplant, indices_remove, layer, depth):

    weakest_weights_copy = copy.deepcopy(weakest_weights)

    for index in range(7):
        if index == 0:
            # order filters
            fittest_weights[index + depth][:, :, :, indices_remove[layer]] = weakest_weights_copy[index + depth][:, :, :,
                                                                             indices_transplant[layer]]
        elif index == [1, 2, 3, 4, 5]:
            # order the biases and the batch norm parameters
            fittest_weights[index + depth][indices_remove[layer]] = weakest_weights_copy[index + depth][
                indices_transplant[layer]]
        elif index == 6:
            if (index + depth) != (len(fittest_weights) - 3):
                # order channels
                fittest_weights[index + depth][:, :, indices_remove[layer], :] = weakest_weights_copy[index + depth][:, :,
                                                                                 indices_transplant[layer], :]
            else:  # this is for the flattened fully connected layer

                num_filters = 64
                activation_map_size = int(weakest_weights_copy[index + depth].shape[0] / num_filters)

                for i in range(len(indices_transplant[layer])):
                    filter_id_transplant = indices_transplant[layer][i]
                    filter_id_remove = indices_remove[layer][i]
                    fittest_weights[index + depth][
                        [num_filters * j + filter_id_remove for j in range(activation_map_size)]] = weakest_weights_copy[index + depth][
                        [num_filters * j + filter_id_transplant for j in range(activation_map_size)]]

    return fittest_weights


def arithmetic_crossover(fittest_weights, weakest_weights, t=0.5):

    array_one = np.array(fittest_weights)
    array_two = np.array(weakest_weights)

    # the scale factor is to keep the same variance
    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))
    new_weights = (t * array_one + (1 - t) * array_two) * scale_factor

    return new_weights


def crossover_method(weights_one, weights_two, list_corr_matrices, crossover):

    list_ordered_indices_one = []
    list_ordered_indices_two = []

    for index in range(len(list_corr_matrices)):
        corr_matrix_nn = list_corr_matrices[index]

        indices_one, indices_two = bipartite_matching(corr_matrix_nn, crossover)
        list_ordered_indices_one.append(indices_one)
        list_ordered_indices_two.append(indices_two)

    weights_nn_one_copy = list(weights_one)
    weights_nn_two_copy = list(weights_two)
    list_ordered_w_one = permute_cnn(weights_nn_one_copy, list_ordered_indices_one)
    list_ordered_w_two = permute_cnn(weights_nn_two_copy, list_ordered_indices_two)

    return list_ordered_indices_one, list_ordered_indices_two, list_ordered_w_one, list_ordered_w_two


def compute_pareto(data):
    sorted_data = sorted(data, key=operator.itemgetter(0, 1), reverse=False)
    pareto_idx = list()
    pareto_idx.append(0)

    cutt_off_fitness = sorted_data[0][0]
    cutt_off_length = sorted_data[0][1]

    for i in range(1, len(sorted_data)):
        if sorted_data[i][0] < cutt_off_fitness or sorted_data[i][1] < cutt_off_length:
            pareto_idx.append(i)
            if sorted_data[i][0] < cutt_off_fitness:
                cutt_off_fitness = sorted_data[i][0]
            else:
                cutt_off_length = sorted_data[i][1]

    return np.array(sorted_data), pareto_idx

