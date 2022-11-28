import tensorflow as tf
import os

from loaders.nga_dataset import nga_dataset_generator
import matplotlib.pyplot as plt

CHECKPOINT_PATH = 'checkpoints/autoencoder.ckpt'
GRAPHS_PATH = 'graphs'
BASE_FILTER_SIZE = 12
IMAGE_GRID_WIDTH = 5
KERNEL_SIZE = 4


class AutoEncoderCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, folder):
        super().__init__()
        self.generator = nga_dataset_generator(folder, batch_size=IMAGE_GRID_WIDTH ** 2)

    @staticmethod
    def plot_and_save(images, path):
        _, axs = plt.subplots(IMAGE_GRID_WIDTH, IMAGE_GRID_WIDTH, figsize=(16, 16))
        axs = axs.flatten()
        for img, ax in zip(images, axs):
            ax.imshow(img)

        plt.savefig(path)
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        images, _ = next(self.generator)

        predictions = self.model.predict(images)
        AutoEncoderCustomCallback.plot_and_save(images, os.path.join(GRAPHS_PATH, '{}_exp.jpg'.format(epoch)))
        AutoEncoderCustomCallback.plot_and_save(predictions, os.path.join(GRAPHS_PATH, '{}_pred.jpg'.format(epoch)))


def build_autoencoder_callbacks(folder):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                    save_weights_only=True,
                                                    monitor='loss', mode='min', save_best_only=True)
    return [checkpoint, AutoEncoderCustomCallback(folder)]


def build_encoder():
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE, kernel_size=KERNEL_SIZE, strides=2, padding='SAME', activation='relu',
                               input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE * 2, kernel_size=KERNEL_SIZE, strides=2, padding='SAME',
                               activation='relu'),
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE * 4, kernel_size=KERNEL_SIZE, strides=2, padding='SAME',
                               activation='relu'),
        tf.keras.layers.Flatten(),
    ])
    return encoder


def build_decoder():
    decoder = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=(4, 4, BASE_FILTER_SIZE * 4), input_shape=(768,)),
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE, kernel_size=KERNEL_SIZE, strides=1, padding='SAME', activation='relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE * 4, kernel_size=KERNEL_SIZE, strides=1, padding='SAME'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(BASE_FILTER_SIZE * 2, kernel_size=KERNEL_SIZE, strides=1, padding='SAME'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(3, kernel_size=KERNEL_SIZE, strides=1, padding='SAME', activation='sigmoid'),
    ])
    return decoder


def build_autoencoder():
    encoder = build_encoder()
    decoder = build_decoder()
    autoencoder = tf.keras.models.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, autoencoder
