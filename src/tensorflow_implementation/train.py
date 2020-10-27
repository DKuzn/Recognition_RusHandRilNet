from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import json
from tensorflow_implementation.dataset import Dataset
from tensorflow_implementation.model import load_model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_function(train_ds, test_ds, epochs=100, steps=2325, val_steps=930):
    model = load_model('../../model/R2HandRilNet.json')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                     beta_1=0.9,
                                                     beta_2=0.999,
                                                     epsilon=1e-07,
                                                     amsgrad=False,
                                                     name='Adam'),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../model/weights/best.h5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='min',
                                                    save_freq='epoch')

    checkpoint_path = '../../model/weights/last.h5'
    train_plot_path = '../../model/train_plot.json'

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    with tf.device('/GPU:0'):
        history = model.fit(train_ds,
                            epochs=epochs,
                            steps_per_epoch=steps,
                            validation_data=test_ds,
                            validation_steps=val_steps,
                            callbacks=[checkpoint])

    model.save_weights(checkpoint_path)

    if os.path.exists(train_plot_path):
        out_result = json.load(open(train_plot_path, 'r'))
    else:
        out_result = [[], [], [], []]

    out_result[0] += history.history['accuracy']
    out_result[1] += history.history['val_accuracy']
    out_result[2] += history.history['loss']
    out_result[3] += history.history['val_loss']
    with open(train_plot_path, 'w') as f:
        json.dump(out_result, f)


if __name__ == '__main__':
    BATCH_SIZE = 100

    train_dataset = Dataset('../../R2HandRilDataset/Train', BATCH_SIZE).load_data()
    test_dataset = Dataset('../../R2HandRilDataset/Test', BATCH_SIZE).load_data()
    train_function(train_dataset, test_dataset, 10)
