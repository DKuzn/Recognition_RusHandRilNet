import tensorflow as tf
import pathlib
import random as rd


class Dataset:
    def __init__(self, path, batch_size):
        self.__AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.__BATCH_SIZE = batch_size
        self.__data_root_orig = path
        self.__data_root = pathlib.Path(self.__data_root_orig)
        self.__all_image_paths = self.__image_paths()
        self.__image_count = len(self.__all_image_paths)
        self.__all_image_labels = self.__get_image_labels()

    def __image_paths(self):
        image_paths = list(self.__data_root.glob('*/*'))
        image_paths = [str(path) for path in image_paths]
        rd.shuffle(image_paths)
        return image_paths

    def get_labels(self):
        return sorted(item.name for item in self.__data_root.glob('*/') if item.is_dir())

    def __get_image_labels(self):
        label_names = self.get_labels()
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in self.__all_image_paths]
        return all_image_labels

    @staticmethod
    def __preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [28, 28])
        image /= 255.0  # normalize to [0,1] range
        return image

    def __load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.__preprocess_image(image)

    def __pack_dataset(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.__all_image_paths)
        image_ds = path_ds.map(self.__load_and_preprocess_image, num_parallel_calls=self.__AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.__all_image_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = image_label_ds.shuffle(buffer_size=self.__image_count)
        ds = ds.repeat()
        ds = ds.batch(self.__BATCH_SIZE)
        ds = ds.prefetch(buffer_size=self.__AUTOTUNE)
        return ds

    def load_data(self):
        return self.__pack_dataset()
