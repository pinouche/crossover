import keras


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, epoch_list, parent_id, work_id):
        self.epoch_list = epoch_list
        self.parent_id = parent_id
        self.work_id = work_id

    def on_epoch_end(self, epoch, logs={}):
        if epoch+1 in self.epoch_list:
            self.model.save("parents_trained/model_" + self.parent_id + "_epoch_" + str(epoch+1) + "_" + str(self.work_id) + ".hd5")


def model_keras(seed, data, weights_hidden_size=None):
    if data == "mnist":
        input_size = 784
        hidden_size = 512
        output_size = 10

        if weights_hidden_size is not None:
            hidden_size = weights_hidden_size

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               kernel_initializer=initializer, input_shape=(input_size,)),
            # output layer
            keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                               kernel_initializer=initializer),

            keras.layers.Activation(keras.activations.softmax)
        ])

    elif data == "cifar10":
        input_size = 3072
        hidden_size = 100
        output_size = 10

        if weights_hidden_size is not None:
            hidden_size = weights_hidden_size

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               kernel_initializer=initializer, input_shape=(input_size,)),

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               kernel_initializer=initializer),

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               kernel_initializer=initializer),
            # output layer
            keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                               kernel_initializer=initializer),

            keras.layers.Activation(keras.activations.softmax)
        ])

    else:
        raise Exception("wrong dataset")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model
