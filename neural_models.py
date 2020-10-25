import keras


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, epoch_list, parent_id, work_id):
        self.epoch_list = epoch_list
        self.parent_id = parent_id
        self.work_id = work_id

    def on_epoch_end(self, epoch, logs={}):
        if epoch+1 in self.epoch_list:
            self.model.save("parents_trained/model_" + self.parent_id + "_epoch_" + str(epoch+1) + "_" + str(self.work_id) + ".hd5")


def linear_classifier_keras(seed, input_size, data):

    # for mnist and cifar10
    output_size = 10
    if data == "cifar100":
        output_size = 20

    initializer = keras.initializers.glorot_normal(seed=seed)

    model = keras.models.Sequential([

        keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                           trainable=True, kernel_initializer=initializer, input_shape=(input_size,)),

        keras.layers.Activation(keras.activations.softmax)
    ])

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model


def model_keras(seed, data, trainable_list=[]):

    if data == "mnist":
        input_size = 784
        hidden_size = 512
        output_size = 10
        num_layers = 2

        if len(trainable_list) == 0:
            trainable_list = [True] * num_layers

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               trainable=trainable_list[0], kernel_initializer=initializer, input_shape=(input_size,)),
            # output layer
            keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                               trainable=trainable_list[1], kernel_initializer=initializer),

            keras.layers.Activation(keras.activations.softmax)
        ])

    elif data == "cifar10" or data == "cifar100":
        input_size = 3072
        hidden_size = 256
        num_layers = 4

        output_size = 10
        if data == "cifar100":
            output_size = 20

        if len(trainable_list) == 0:
            trainable_list = [True] * num_layers

        initializer = keras.initializers.glorot_normal(seed=seed)

        model = keras.models.Sequential([

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               trainable=trainable_list[0], kernel_initializer=initializer, input_shape=(input_size,)),

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               trainable=trainable_list[1], kernel_initializer=initializer),

            keras.layers.Dense(hidden_size, activation=keras.activations.selu, use_bias=True,
                               trainable=trainable_list[2], kernel_initializer=initializer),
            # output layer
            keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                               trainable=trainable_list[3], kernel_initializer=initializer),

            keras.layers.Activation(keras.activations.softmax)
        ])

    else:
        raise Exception("wrong dataset")

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model


def keras_model_cnn(seed, data, trainable_list=[]):

    num_trainable_layers = 7
    output_size = 10
    if data == "cifar100":
        output_size = 20

    if len(trainable_list) == 0:
        trainable_list = [True] * num_trainable_layers

    initializer = keras.initializers.glorot_normal(seed=seed)

    model = keras.models.Sequential([

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[0], padding='same', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[1], padding='same'),

        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[2], padding='same', input_shape=(32, 32, 3)),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[3], padding='same'),

        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[4], padding='same', input_shape=(32, 32, 3)),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            trainable=trainable_list[5], padding='same'),

        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),

        # output layer
        keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                           trainable=trainable_list[6], kernel_initializer=initializer),

        keras.layers.Activation(keras.activations.softmax)
    ])

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model
