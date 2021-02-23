import keras


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, epoch_list, parent_id, work_id):
        self.epoch_list = epoch_list
        self.parent_id = parent_id
        self.work_id = work_id

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 in self.epoch_list:
            self.model.save("parents_trained/model_" + self.parent_id + "_epoch_" + str(epoch + 1) + "_" + str(
                self.work_id) + ".hd5")


def lr_scheduler(epoch, learning_rate=0.1, lr_drop=20):
    new_lr = learning_rate * (0.5 ** (epoch // lr_drop))

    return new_lr


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


def keras_model_cnn(seed, data, trainable_list=[]):

    num_trainable_layers = 5
    input_shape = (32, 32, 3)
    output_size = 10
    if data == "cifar100":
        output_size = 20

    if len(trainable_list) == 0:
        trainable_list = [True] * (num_trainable_layers*2-1)

    initializer = keras.initializers.glorot_normal(seed=seed)

    model = keras.models.Sequential([

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            padding='same', input_shape=input_shape, trainable=trainable_list[0]),
        keras.layers.BatchNormalization(momentum=0.9, trainable=trainable_list[1]),
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            padding='same', trainable=trainable_list[2]),
        keras.layers.BatchNormalization(momentum=0.9, trainable=trainable_list[3]),

        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            padding='same', trainable=trainable_list[4]),
        keras.layers.BatchNormalization(momentum=0.9, trainable=trainable_list[5]),

        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer,
                            padding='same', trainable=trainable_list[6]),
        keras.layers.BatchNormalization(momentum=0.9, trainable=trainable_list[7]),

        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),

        # output layer
        keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                           kernel_initializer=initializer, trainable=trainable_list[8]),

        keras.layers.Activation(keras.activations.softmax)
    ])

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model


# Build the network of vgg-16 with dropout and weight decay as described in the original paper.
def keras_vgg(seed, data):
    input_shape = (32, 32, 3)
    weight_decay = 0.0005
    initializer = keras.initializers.glorot_normal(seed=seed)

    output_size = 10
    if data == "cifar100":
        output_size = 20

    model = keras.models.Sequential([

        keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay),
                           kernel_initializer=initializer),
        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        # output layer
        keras.layers.Dense(output_size, activation=keras.activations.linear, use_bias=False,
                           kernel_initializer=initializer),
        keras.layers.Activation(keras.activations.softmax)

    ])

    # optimizer
    learning_rate = 0.1
    lr_decay = 1e-6
    optimizer = keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])

    return model
