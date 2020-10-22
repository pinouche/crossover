import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
import keras
from keras.models import load_model


# load data functions
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    return x_train, x_test, y_train, y_test


def load_cifar_100(flatten=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="coarse")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], 3072))
        x_test = np.reshape(x_test, (x_test.shape[0], 3072))

    return x_train, x_test, y_train, y_test


def load_cifar(flatten=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], 3072))
        x_test = np.reshape(x_test, (x_test.shape[0], 3072))

    return x_train, x_test, y_train, y_test


def partition_classes(x_train, x_test, y_train, y_test):

    cut_off = int(len(np.unique(y_train))/4)

    mask_train = np.squeeze(y_train >= cut_off)
    mask_test = np.squeeze(y_test >= cut_off)

    x_train_s1 = x_train[mask_train]
    y_train_s1 = y_train[mask_train]
    x_test_s1 = x_test[mask_test]
    y_test_s1 = y_test[mask_test]

    x_train_s2 = x_train[~mask_train]
    y_train_s2 = y_train[~mask_train]
    x_test_s2 = x_test[~mask_test]
    y_test_s2 = y_test[~mask_test]

    return x_train_s1, y_train_s1, x_test_s1, y_test_s1, x_train_s2, y_train_s2, x_test_s2, y_test_s2


def add_noise(parent_weights, seed, t=0.5, sensitivity_vector=None, scaling_mutation_method=[]):
    upper_limit = 2
    lower_limit = 1 / upper_limit

    mutation_scaling = 1
    if "safe" in scaling_mutation_method:
        sensitivity_vector = (upper_limit - lower_limit) * ((sensitivity_vector - np.min(sensitivity_vector)) /
                                                            (np.max(sensitivity_vector) - np.min(
                                                                sensitivity_vector))) + lower_limit
        mutation_scaling = 1 / sensitivity_vector
    elif "unsafe" in scaling_mutation_method:
        sensitivity_vector = (upper_limit - lower_limit) * ((sensitivity_vector - np.min(sensitivity_vector)) /
                                                            (np.max(sensitivity_vector) - np.min(
                                                                sensitivity_vector))) + lower_limit
        mutation_scaling = sensitivity_vector

    np.random.seed(seed)

    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))
    mean_parent, std_parent = 0.0, np.std(parent_weights)

    weight_noise = np.random.normal(loc=mean_parent, scale=std_parent, size=parent_weights.shape)
    weight_noise = weight_noise * mutation_scaling
    parent_weights = (t * parent_weights + (1 - t) * weight_noise) * scale_factor

    return parent_weights


def get_hidden_layers(model, data_x):
    def keras_function_layer(model_layer, data):
        hidden_func = keras.backend.function(model.layers[0].input, model_layer.output)
        result = hidden_func([data])

        return result

    hidden_layers_list = []
    for index in range(len(model.layers) - 2):
        hidden_layer = keras_function_layer(model.layers[index], data_x)
        hidden_layers_list.append(hidden_layer)

    return hidden_layers_list


def get_gradients_hidden_layers(model, data_x, data_y):
    y_true = keras.Input(shape=(1,))
    loss = keras.backend.categorical_crossentropy(y_true, model.output)
    model_layers = [layer.output for layer in model.layers[:-2]]

    gradients = keras.backend.gradients(loss, model_layers)
    func = keras.backend.function([model.inputs, y_true], gradients)

    gradient_list = func([data_x, data_y])

    return gradient_list


def get_magnitude_weights(weights_list):
    return np.abs(np.array(weights_list))


def get_movement_weights(weights_list, best_parent_string, work_id):
    best_initial_model = load_model("parents_initial/" + best_parent_string + "_initial_" + str(work_id) + ".hd5")
    best_initial_weights = best_initial_model.get_weights()
    weight_movements = np.abs(np.array(weights_list) - np.array(best_initial_weights))

    return weight_movements


def compute_neurons_variance(activations):
    var_list = []

    if isinstance(activations, list):

        for layer in range(len(activations)):
            activation_matrix = activations[layer]
            neurons_variance = np.var(activation_matrix, axis=0)
            var_list.append(neurons_variance)
    else:
        neurons_variance = np.var(activations, axis=0)
        var_list.append(neurons_variance)

    return var_list


def identify_interesting_neurons(mask_convergence_best_parent, mask_convergence_worst_parent, list_corr_matrices,
                                 corr_wanted="low"):
    neurons_indices_list_worst_parent = []
    indices_neurons_non_converged_best_parent_list = []

    # want those that have converged in worst parent
    mask_convergence_worst_parent = [~mask for mask in mask_convergence_worst_parent]

    for index in range(len(list_corr_matrices)):

        mask_best_parent = mask_convergence_best_parent[index]
        mask_worst_parent = mask_convergence_worst_parent[index]
        number_of_neurons_to_replace = min(np.sum(mask_best_parent), np.sum(mask_worst_parent))
        indices_neurons_non_converged_best_parent = [index for index in range(len(mask_best_parent)) if
                                                     mask_best_parent[index]]

        # choose randomly the indices of neurons to replace from the non-converged neurons in the best parent
        if corr_wanted == "low":
            indices_neurons_non_converged_best_parent = indices_neurons_non_converged_best_parent[
                                                        :number_of_neurons_to_replace]
        elif corr_wanted == "high":
            indices_neurons_non_converged_best_parent = indices_neurons_non_converged_best_parent[
                                                        -number_of_neurons_to_replace:]

        indices_neurons_non_converged_best_parent_list.append(indices_neurons_non_converged_best_parent)

        corr_matrix = list_corr_matrices[index][~mask_best_parent]

        max_corr_list = []
        for j in range(corr_matrix.shape[1]):
            if mask_worst_parent[j]:
                corr = np.abs(corr_matrix[:, j])
                max_corr = np.max(corr)
                max_corr_list.append((max_corr, j))
        max_corr_list.sort()
        print(max_corr_list)
        indices = [tuples[1] for tuples in max_corr_list[:number_of_neurons_to_replace]]
        neurons_indices_list_worst_parent.append(indices)

    return neurons_indices_list_worst_parent, indices_neurons_non_converged_best_parent_list


def compute_mask_convergence(variance, q):
    mask_list = []

    if isinstance(variance, list):

        for layer in range(len(variance)):
            var_layer = variance[layer]
            percentile_value = sorted(var_layer)[:int(len(var_layer) * q + 0.5)][-1]
            mask_layer = var_layer < percentile_value

            mask_list.append(mask_layer)
    else:
        percentile_value = sorted(variance)[:int(len(variance) * q + 0.5)][-1]
        mask_layer = variance <= percentile_value

        mask_list.append(mask_layer)

    return mask_list


def get_gradient_weights(model, data_x):
    batch_size = 2048
    loss = model.layers[-2].output

    trainable_weights_list = model.trainable_weights
    gradients = keras.backend.gradients(loss, trainable_weights_list)
    get_gradients = keras.backend.function(model.inputs, gradients)

    gradient_list = []
    for index in range(batch_size):
        gradient_list.append(get_gradients([np.expand_dims(data_x[index, :], axis=0)]))

    gradient_list = np.mean(np.abs(np.array(gradient_list)), axis=0)

    return gradient_list


def get_corr(hidden_representation_list_one, hidden_representation_list_two):
    list_corr_matrices = []

    for index in range(len(hidden_representation_list_one)):
        hidden_representation_one = hidden_representation_list_one[index]
        hidden_representation_two = hidden_representation_list_two[index]

        n = hidden_representation_one.shape[1]

        corr_matrix_nn = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                corr = np.corrcoef(hidden_representation_one[:, i], hidden_representation_two[:, j])[0, 1]
                corr_matrix_nn[i, j] = corr

        list_corr_matrices.append(corr_matrix_nn)

    return list_corr_matrices


# cross correlation function for both bipartite matching (hungarian method)
def bipartite_matching(corr_matrix_nn, crossover="unsafe_crossover"):
    if crossover == "unsafe_crossover":
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "safe_crossover":
        corr_matrix_nn *= -1  # default of linear_sum_assignement is to minimize cost, we want to max "cost"
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "orthogonal_crossover":
        corr_matrix_nn = np.abs(corr_matrix_nn)
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "normed_crossover":
        corr_matrix_nn = np.abs(corr_matrix_nn)
        corr_matrix_nn *= -1
        list_neurons_x, list_neurons_y = linear_sum_assignment(corr_matrix_nn)  # Hungarian method
    elif crossover == "naive_crossover":
        list_neurons_x, list_neurons_y = list(range(corr_matrix_nn.shape[0])), list(range(corr_matrix_nn.shape[0]))
    else:
        raise ValueError('the crossover method is not defined')

    return list_neurons_x, list_neurons_y


def get_network_similarity(list_corr_matrices, list_ordered_indices_one, list_ordered_indices_two):
    list_meta = []

    for layer_num in range(len(list_corr_matrices)):
        list_corr = []
        for index in range(len(list_ordered_indices_one)):
            i = list_ordered_indices_one[layer_num][index]
            j = list_ordered_indices_two[layer_num][index]
            corr = np.abs(list_corr_matrices[layer_num][i][j])
            list_corr.append(corr)

        list_meta.append(np.mean(list_corr))

    similarity = np.mean(list_meta)

    return similarity


# Algorithm 2
def apply_mask_to_weights(nn_weights_list, list_indices_hidden):
    count = 0
    depth = count * 2
    for layer in range(len(list_indices_hidden)):
        for index in range(3):
            if index == 0:
                # order columns for weights
                nn_weights_list[index + depth] = nn_weights_list[index + depth][:, list_indices_hidden[layer]]
            elif index == 1:
                nn_weights_list[index + depth] = nn_weights_list[index + depth][
                    list_indices_hidden[layer]]  # order columns for bias
            elif index == 2:
                # order rows
                nn_weights_list[index + depth] = nn_weights_list[index + depth][list_indices_hidden[layer], :]

        count += 1
        depth = count * 2

    nn_weights_list = np.asarray(nn_weights_list)

    return nn_weights_list


def apply_mask_and_add_noise(nn_weights_list, list_mask, seed):
    count = 0
    depth = count * 2
    for layer in range(len(list_mask)):
        for index in range(3):
            if index == 0:
                # noise the columns
                nn_weights_list[index + depth][:, list_mask[layer]] = \
                    add_noise(nn_weights_list[index + depth][:, list_mask[layer]], seed, 0.5)
            elif index == 1:
                # order columns for bias
                nn_weights_list[index + depth][list_mask[layer]] = \
                    add_noise(nn_weights_list[index + depth][list_mask[layer]], seed, 0.5)
            elif index == 2:
                # order rows
                nn_weights_list[index + depth][list_mask[layer], :] = \
                    add_noise(nn_weights_list[index + depth][list_mask[layer], :], seed, 0.5)

        count += 1
        depth = count * 2

    nn_weights_list = np.asarray(nn_weights_list)

    return nn_weights_list


def transplant_neurons(fittest_weights, weakest_weights, indices_neurons_transplant, indices_neurons_to_remove,
                       layer, depth):
    for index in range(3):
        if index == 0:
            # order the columns
            fittest_weights[index + depth][:, indices_neurons_to_remove[layer]] = \
                weakest_weights[index + depth][:, indices_neurons_transplant[layer]]
        elif index == 1:
            # order columns for bias
            fittest_weights[index + depth][indices_neurons_to_remove[layer]] = \
                weakest_weights[index + depth][indices_neurons_transplant[layer]]
        elif index == 2:
            # order rows
            fittest_weights[index + depth][indices_neurons_to_remove[layer], :] = \
                weakest_weights[index + depth][indices_neurons_transplant[layer], :]

    return fittest_weights


def crossover_method(weights_one, weights_two, list_corr_matrices, crossover):
    list_ordered_indices_one = []
    list_ordered_indices_two = []
    for index in range(len(list_corr_matrices)):
        corr_matrix_nn = list_corr_matrices[index]

        indices_one, indices_two = bipartite_matching(corr_matrix_nn, crossover)
        list_ordered_indices_one.append(indices_one)
        list_ordered_indices_two.append(indices_two)

    similarity = get_network_similarity(list_corr_matrices, list_ordered_indices_one, list_ordered_indices_two)

    # order the weight matrices

    if crossover == "naive":
        list_ordered_w_one = list(weights_one)
        list_ordered_w_two = list(weights_two)

    else:
        weights_nn_one_copy = list(weights_one)
        weights_nn_two_copy = list(weights_two)
        list_ordered_w_one = apply_mask_to_weights(weights_nn_one_copy, list_ordered_indices_one)
        list_ordered_w_two = apply_mask_to_weights(weights_nn_two_copy, list_ordered_indices_two)

    return list_ordered_w_one, list_ordered_w_two, similarity


def arithmetic_crossover(network_one, network_two, t=0.5):
    scale_factor = np.sqrt(1 / (np.power(t, 2) + np.power(1 - t, 2)))

    list_weights = []
    for index in range(len(network_one)):
        averaged_weights = (t * network_one[index] + (1 - t) * network_two[index]) * scale_factor
        list_weights.append(averaged_weights)

    return list_weights


def add_noise_to_fittest(best_parent, crossover, seed, sensitivity_vector=None, safe_mutation=[]):
    t = 0.5
    if crossover == "noise_0.1":
        t = 0.9

    mutation_scaling = None

    # choose best parent
    list_weights = []
    for index in range(len(best_parent)):
        if sensitivity_vector is not None:
            mutation_scaling = sensitivity_vector[index]
        parent_weights = best_parent[index]
        parent_weights = add_noise(parent_weights, seed, t, mutation_scaling, safe_mutation)

        list_weights.append(parent_weights)

    return list_weights


def corr_neurons(fittest_weights, corr_matrices_list, seed, crossover, threshold):
    mask_list = []
    for corr_matrix in corr_matrices_list:
        max_corr_list = []
        for i in range(corr_matrix.shape[0]):
            corr = np.abs(corr_matrix[i, :])
            corr.sort()
            max_corr_list.append(corr[-2])

        if crossover == "noise_low_corr":
            mask_array = np.asarray(max_corr_list) <= np.quantile(max_corr_list, 1 - threshold)
        elif crossover == "noise_high_corr":
            mask_array = np.asarray(max_corr_list) >= np.quantile(max_corr_list, threshold)

        mask_list.append(mask_array)

    list_weights = apply_mask_and_add_noise(fittest_weights, mask_list, seed)

    return list_weights


def scale_fittest_parent(fittest_weights):
    fittest_weights = fittest_weights / np.sqrt(2)

    return fittest_weights


def load_model_long_string(best_initial_id, work_id, epoch):
    model = load_model(
        "parents_trained/model_" + best_initial_id + "_epoch_" + str(epoch + 1) + "_" + str(work_id) + ".hd5")

    return model
