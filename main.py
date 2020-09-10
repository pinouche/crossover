import numpy as np
from timeit import default_timer as timer
import threading
import queue
import multiprocessing
import warnings
import pickle

from keras.models import load_model
import keras

from utils import get_gradients_hidden_layers
from utils import get_hidden_layers
from utils import get_gradient_weights
from utils import get_corr
from utils import crossover_method
from utils import arithmetic_crossover
from utils import load_cifar
from utils import load_mnist
from utils import add_noise_to_fittest
from utils import corr_neurons

from feed_forward import CustomSaver
from feed_forward import model_keras

warnings.filterwarnings("ignore")


def crossover_offspring(data, x_train, y_train, x_test, y_test, pair_list, work_id, data_struc,
                        parallel="process"):
    # shuffle input data here
    # np.random.seed(work_id)
    # shuffle_list = np.arange(x_train.shape[0])
    # np.random.shuffle(shuffle_list)
    # x_train = x_train[shuffle_list]
    # y_train = y_train[shuffle_list]

    num_pairs = len(pair_list)

    print("FOR PAIR NUMBER " + str(work_id + 1))

    # crossover_types = ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover",
    # "naive_crossover", # "noise_low_corr", "noise_high_corr", "safe_mutation", "unsafe_mutation"]
    crossover_types = ["safe_mutation"]

    vector_representation = "gradient"  # "gradient" or "activation"
    result_list = [[] for _ in range(len(crossover_types)+1)]
    quantile = 0.5
    total_training_epoch = 25
    epoch_list = [20]

    model_one = model_keras(work_id, data)
    model_two = model_keras(work_id + num_pairs, data)
    model_one.save("parent_one_initial")
    model_two.save("parent_two_initial")

    print("one")
    save_callback = CustomSaver(epoch_list, "parent_one")
    model_information_parent_one = model_one.fit(x_train, y_train, epochs=total_training_epoch, batch_size=256,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])

    save_callback = CustomSaver(epoch_list, "parent_two")
    model_information_parent_two = model_two.fit(x_train, y_train, epochs=total_training_epoch, batch_size=256,
                                                 verbose=False,
                                                 validation_data=(x_test, y_test), callbacks=[save_callback])
    print("three")

    count = 0
    for crossover in crossover_types:

        # get the parents' weights at the best epoch
        best_epoch_parent_one = np.argmin(model_information_parent_one.history["val_loss"])
        best_epoch_parent_one = np.argmin([np.abs(best_epoch_parent_one - epoch_num) for epoch_num in epoch_list])
        best_epoch_parent_one = epoch_list[best_epoch_parent_one]
        parent_one = load_model("model_parent_one_epoch_" + str(best_epoch_parent_one) + ".hd5")
        weights_nn_one = parent_one.get_weights()

        best_epoch_parent_two = np.argmin(model_information_parent_two.history["val_loss"])
        best_epoch_parent_two = np.argmin([np.abs(best_epoch_parent_two - epoch_num) for epoch_num in epoch_list])
        best_epoch_parent_two = epoch_list[best_epoch_parent_two]
        parent_two = load_model("model_parent_two_epoch_" + str(best_epoch_parent_two) + ".hd5")
        weights_nn_two = parent_two.get_weights()

        fittest_weights, fittest_model = weights_nn_one, parent_one
        loss_best_parent = model_information_parent_one.history["val_loss"]
        if np.min(loss_best_parent) > np.max(model_information_parent_two.history["val_loss"]):
            loss_best_parent = model_information_parent_two.history["val_loss"]
            fittest_weights, fittest_model = weights_nn_two, parent_two

        print("crossover method: " + crossover)
        list_ordered_weights_one, list_ordered_weights_two = weights_nn_one, weights_nn_two

        if vector_representation == "activation":
            list_hidden_representation_one = get_hidden_layers(parent_one, x_test)  # activation vector network one
            list_hidden_representation_two = get_hidden_layers(parent_two, x_test)  # activation vector network two
        elif vector_representation == "gradient":
            list_hidden_representation_one = get_gradients_hidden_layers(parent_one, x_test, y_test)  # gradient vector
            list_hidden_representation_two = get_gradients_hidden_layers(parent_two, x_test, y_test)  # gradient vector

        list_corr_matrices = get_corr(list_hidden_representation_one, list_hidden_representation_two)

        if crossover in ["safe_crossover", "unsafe_crossover", "orthogonal_crossover", "normed_crossover", "naive_crossover"]:
            list_ordered_weights_one, list_ordered_weights_two, parents_similarity = crossover_method(
                list_ordered_weights_one, list_ordered_weights_two, list_corr_matrices, crossover)

        if crossover in ["noise_0.5", "noise_0.1"]:
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id)

        elif crossover in ["safe_mutation", "unsafe_mutation"]:
            sensitivity_gradient_vector = get_gradient_weights(parent_two, x_test)
            weights_crossover = add_noise_to_fittest(fittest_weights, crossover, work_id, sensitivity_gradient_vector,
                                                     True)

        elif crossover in ["noise_low_corr", "noise_high_corr"]:
            weights_crossover = corr_neurons(list_ordered_weights_one, list_ordered_weights_two,
                                             list_corr_matrices, model_information_parent_one,
                                             model_information_parent_two, work_id, best_epoch_parent_one,
                                             crossover, quantile)
        else:
            weights_crossover = arithmetic_crossover(list_ordered_weights_one, list_ordered_weights_two)

        model_offspring = model_keras(0, data, weights_crossover[0].shape[1])

        model_offspring.set_weights(weights_crossover)
        model_information_offspring = model_offspring.fit(x_train, y_train,
                                                          epochs=total_training_epoch,
                                                          batch_size=256,
                                                          verbose=False, validation_data=(x_test, y_test))

        if count == 0:
            result_list[count].append(loss_best_parent)
        result_list[count+1].append(model_information_offspring.history["val_loss"])

        keras.backend.clear_session()
        count += 1

    if parallel == "process":
        data_struc[str(work_id) + "_performance"] = result_list
    elif parallel == "thread":
        data_struc.put(result_list)

    print("ten")


if __name__ == "__main__":

    data = "cifar10"

    if data == "cifar10":
        x_train, x_test, y_train, y_test = load_cifar()
    elif data == "mnist":
        x_train, x_test, y_train, y_test = load_mnist()

    parallel_method = "process"

    if parallel_method == "thread":
        num_threads = 1

        start = timer()

        q = queue.Queue()

        pair_list = [pair for pair in range(num_threads)]

        t = [threading.Thread(target=crossover_offspring, args=(data, x_train, y_train, x_test, y_test,
                                                                pair_list, i, q, parallel_method)) for i in
             range(num_threads)]

        for thread in t:
            thread.start()

        results = [q.get() for _ in range(num_threads)]

        # Stop these threads
        for thread in t:
            thread.stop = True

        end = timer()
        print(end - start)

    elif parallel_method == "process":
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
