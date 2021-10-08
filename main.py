import numpy as np
from timeit import default_timer as timer
import warnings
import pickle
import copy

import tensorflow as tf
import keras

from sklearn.metrics import accuracy_score

from load_data import load_cifar
from load_data import load_mnist
from load_data import load_cifar_100
from load_data import add_negative_class_examples
from load_data import shift_labels

from utils import transplant_neurons
from utils import align_neurons

from neural_models import keras_model_cnn

warnings.filterwarnings("ignore")


def transplant_crossover(crossover, data_main, data_subset, data_full, class_list, num_transplants=1, num_conv_layer=5, batch_size_activation=512,
                         batch_size_sgd=128, work_id=0):
    dic_results = dict()
    print("crossover method: " + crossover)
    for safety_level in ["safe_crossover", "naive_crossover"]:
        print(safety_level)

        num_classes_main, num_classes_subset = len(np.unique(data_main[1])), len(np.unique(data_subset[1]))

        model_main = keras_model_cnn(work_id, num_classes_main)
        model_subset = keras_model_cnn(work_id + 1000, num_classes_subset)

        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model_main.fit(data_main[0], data_main[1], batch_size=batch_size_sgd, epochs=2,
                       verbose=0, validation_data=(data_main[2], data_main[3]), callbacks=[early_stop_callback])

        model_subset.fit(data_subset[0], data_subset[1], batch_size=batch_size_sgd, epochs=2,
                         verbose=0, validation_data=(data_subset[2], data_subset[3]), callbacks=[early_stop_callback])

        # proportion of filters to transfer: we set it to the subset class proportion. We can experiment with pareto using variance also, later.
        # num_swap = (len(np.unique(data_subset[1])) / (len(np.unique(data_main[1])) + len(np.unique(data_subset[1])))) * list_cross_corr[0].shape[0]
        num_swap = int(10)

        for epoch in range(num_transplants):

            weights_main = model_main.get_weights()
            weights_subset = model_subset.get_weights()

            weights_main_tmp = copy.deepcopy(weights_main)
            weights_subset_tmp = copy.deepcopy(weights_subset)

            depth = 0
            for layer in range(num_conv_layer):

                weights_main, weights_subset, neurons_to_remove_main, neurons_to_transplant_main = align_neurons(weights_main, weights_subset,
                                                                                                                 model_main, model_subset, x_test,
                                                                                                                 batch_size_activation,
                                                                                                                 num_conv_layer, num_swap,
                                                                                                                 safety_level, crossover)

                # transplant offspring one
                weights_main = transplant_neurons(weights_main, weights_subset_tmp, neurons_to_transplant_main, neurons_to_remove_main, layer, depth)

                model_main = keras_model_cnn(0, num_classes_main)
                model_subset = keras_model_cnn(0, num_classes_subset)
                model_main.set_weights(weights_main)
                model_subset.set_weights(weights_subset)

                depth = (layer + 1) * 6

        # instantiate new randomly init weights
        model_main = keras_model_cnn(work_id, len(np.unique(data_full[1])), 0.0001)
        random_init_weights = model_main.get_weights()

        print("COMPUTING FOR RESETING FILTERS TO RANDOM INIT")

        weight_random_init_filters = copy.deepcopy(weights_main_tmp)

        for index in range(num_conv_layer):
            # layer = weight_random_init_filters[index * 6]
            # filter_indices = np.array(range(layer.shape[-1]))
            # indices = np.random.choice(filter_indices, num_swap, replace=False)

            indices = neurons_to_remove_main[index]
            weight_random_init_filters[index * 6][:, :, :, indices] = random_init_weights[index * 6][:, :, :, indices]

        # reset upper layers to random initialization
        weight_random_init_filters[-3:] = random_init_weights[-3:]
        model_main.set_weights(weight_random_init_filters)

        random_reset_zero_epoch_loss = model_main.evaluate(data_full[2], data_full[3])

        # train the newly transplanted network
        info_random_reset = model_main.fit(data_full[0], data_full[1], batch_size=batch_size_sgd, epochs=50,
                       verbose=2, validation_data=(data_full[2], data_full[3]), callbacks=[early_stop_callback])

        preds = model_main.predict(data_full[2][data_full[3] == class_list[0]])
        print(preds)

        random_reset_loss = info_random_reset.history["val_loss"]
        random_reset_loss.insert(0, random_reset_zero_epoch_loss)
        dic_results["random_reset_method"] = random_reset_loss

        print("COMPUTING FOR TRANSPLANTING METHOD")

        weights_transplant = copy.deepcopy(weights_main)

        # reset upper layers to random initialization
        weights_transplant[-3:] = random_init_weights[-3:]

        # set the weights with the reset last layer
        model_main.set_weights(weights_transplant)
        transplanting_method_zero_epoch_loss = model_main.evaluate(data_full[2], data_full[3])

        # train the newly transplanted network
        info_transplant_loss = model_main.fit(data_full[0], data_full[1], batch_size=batch_size_sgd, epochs=50,
                                                verbose=2, validation_data=(data_full[2], data_full[3]), callbacks=[early_stop_callback])

        transplant_loss = info_transplant_loss.history["val_loss"]
        transplant_loss.insert(0, transplanting_method_zero_epoch_loss)
        dic_results["transplant_method"] = transplant_loss

        print("COMPUTING FOR BASELINE TRANSFER LEARNING METHOD")

        # weights_main_tmp are the weights before transplant
        weights_main_tmp[-3:] = random_init_weights[-3:]

        model_main.set_weights(weights_main_tmp)
        baseline_transfer_learning_zero_epoch_loss = model_main.evaluate(data_full[2], data_full[3])

        # train the newly transplanted network
        info_baseline_transfer = model_main.fit(data_full[0], data_full[1], batch_size=batch_size_sgd, epochs=50,
                                                             verbose=2, validation_data=(data_full[2], data_full[3]), callbacks=[early_stop_callback])

        baseline_transfer_loss = info_baseline_transfer.history["val_loss"]
        baseline_transfer_loss.insert(0, baseline_transfer_learning_zero_epoch_loss)
        dic_results["baseline_transfer_method"] = baseline_transfer_loss
        
        print("COMPUTING ON FULL DATASET FROM SCRATCH")

        model_main.set_weights(random_init_weights)
        training_from_scratch_zero_epoch_loss = model_main.evaluate(data_full[2], data_full[3])
        
        info_scratch_training = model_main.fit(data_full[0], data_full[1], batch_size=batch_size_sgd, epochs=50,
                                                             verbose=2, validation_data=(data_full[2], data_full[3]), callbacks=[early_stop_callback])

        scratch_training_loss = info_scratch_training.history["val_loss"]
        scratch_training_loss.insert(0, training_from_scratch_zero_epoch_loss)
        dic_results["scratch_training_method"] = scratch_training_loss

        keras.backend.clear_session()

    return dic_results


def crossover_offspring(data_main, data_subset, data_full, class_list, work_id=0):
    np.random.seed(work_id + 1)

    # program hyperparameters
    num_conv_layer = 4
    batch_size_activation = 512  # batch_size to compute the activation maps
    batch_size_sgd = 64

    num_transplants = 1

    crossover = "targeted_crossover_variance"

    result_list = transplant_crossover(crossover, data_main, data_subset, data_full, class_list, num_transplants, num_conv_layer,
                                       batch_size_activation, batch_size_sgd, work_id)

    return result_list


if __name__ == "__main__":
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    
    for gpu_device in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_device, True)

    data = "cifar100"
    num_runs = 10

    if data == "cifar10":
        x_train, x_test, y_train, y_test = load_cifar()
    elif data == "cifar100":
        x_train, x_test, y_train, y_test = load_cifar_100()
    else:
        x_train, x_test, y_train, y_test = load_mnist()

    # specify the classes that are to be trained separately
    class_subset_list = [0]

    mask_train = np.array([value in class_subset_list for value in y_train])
    mask_test = np.array([value in class_subset_list for value in y_test])

    data_main = [x_train[~mask_train], y_train[~mask_train], x_test[~mask_test], y_test[~mask_test]]
    data_subset = [x_train[mask_train], y_train[mask_train], x_test[mask_test], y_test[mask_test]]

    data_subset = add_negative_class_examples(copy.deepcopy(data_main), copy.deepcopy(data_subset), class_subset_list, True)
    data_subset = add_negative_class_examples(copy.deepcopy(data_main), copy.deepcopy(data_subset), class_subset_list, False)

    # make the label start from 0 and increment by 1 to train tf model
    data_main[1], data_main[3] = shift_labels(data_main[1], data_main[3])
    data_subset[1], data_subset[3] = shift_labels(data_subset[1], data_subset[3])

    data_full = x_train, y_train, x_test, y_test

    start = timer()

    list_results = []
    for run in range(num_runs):
        results = crossover_offspring(data_main, data_subset, data_full, class_subset_list, run)
        list_results.append(results)

    pickle.dump(list_results, open("crossover_results.pickle", "wb"))

    end = timer()
    print(end - start)
