from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def save_model(model):
    model_json = model.to_json()
    with open("../../model/R2HandRilNet.json", "w") as json_file:
        json_file.write(model_json)


def load_model(path_to_json):
    json_file = open(path_to_json, 'r')
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    return model


def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=124,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='valid',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(filters=124,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='valid'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=596,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=596,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='valid'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=1192,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding='valid'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1192, activation='relu'))
    model.add(tf.keras.layers.Dense(62, activation='softmax'))

    save_model(model)


if __name__ == '__main__':
    define_model()
