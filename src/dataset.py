import tensorflow as tf
import pathlib
import random as rd


class Dataset:
    def __init__(self):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.BATCH_SIZE = None
        self.data_root_orig = '/content/Recognition_RusHandRilNet/Dataset'
        self.data_root = pathlib.Path(self.data_root_orig)
        self.all_image_paths = self.__image_paths()
        self.image_count = len(self.all_image_paths)
        self.all_image_labels = self.__get_image_labels()

    def __image_paths(self):
        all_image_paths = list(self.data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        rd.shuffle(all_image_paths)
        return all_image_paths

    def __get_image_labels(self):
        label_names = sorted(item.name for item in self.data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in self.all_image_paths]
        return all_image_labels

    def __preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        image /= 255.0  # normalize to [0,1] range

        return image

    def __load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.__preprocess_image(image)

    def __pack_dataset(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        image_ds = path_ds.map(self.__load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.all_image_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        # Установка размера буфера перемешивания, равного набору данных, гарантирует
        # полное перемешивание данных.
        ds = image_label_ds.shuffle(buffer_size=self.image_count)
        ds = ds.repeat()
        ds = ds.batch(self.BATCH_SIZE)
        # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        ds = image_label_ds.apply(
          tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count))
        ds = ds.batch(self.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def batch_size(self, batch):
        self.BATCH_SIZE = batch

    def load_data(self):
        # Датасету может понадобиться несколько секунд для старта пока заполняется буфер перемешивания.
        image_batch, label_batch = next(iter(self.__pack_dataset()))
        return image_batch, label_batch
