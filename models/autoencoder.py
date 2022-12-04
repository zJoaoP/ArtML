import tensorflow as tf
import os

from loaders.nga_dataset import nga_dataset_generator
import matplotlib.pyplot as plt

CHECKPOINT_PATH = 'checkpoints/autoencoder.ckpt'
GRAPHS_PATH = 'graphs'
BASE_FILTER_SIZE = 12
IMAGE_GRID_WIDTH = 5
COMPRESSION_SIZE = 32
KERNEL_SIZE = 4
BLOCKS = 3


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
                                                    monitor='val_loss', mode='min', save_best_only=True, verbose=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
    return [checkpoint, early_stopping, AutoEncoderCustomCallback(folder)]


def build_encoder():
    # Build encoder iterating over block count.

    input_layer = x = tf.keras.layers.Input(shape=(32, 32, 3))
    for i in range(BLOCKS):
        factor = 2 ** i
        x = tf.keras.layers.Conv2D(BASE_FILTER_SIZE * factor, kernel_size=KERNEL_SIZE, strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(COMPRESSION_SIZE, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=input_layer, outputs=x)


def build_decoder():
    input_factor = 2 ** (BLOCKS - 1)
    input_width = 32 // (input_factor * 2)

    input_layer = x = tf.keras.layers.Input(shape=(COMPRESSION_SIZE,))

    x = tf.keras.layers.Dense(input_width * input_width * BASE_FILTER_SIZE * input_factor, activation='relu')(x)
    x = tf.keras.layers.Reshape(target_shape=(input_width, input_width, BASE_FILTER_SIZE * input_factor))(x)
    for i in range(BLOCKS):
        factor = 2 ** (BLOCKS - i - 1)
        x = tf.keras.layers.Conv2D(BASE_FILTER_SIZE * factor, kernel_size=KERNEL_SIZE, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(3, kernel_size=KERNEL_SIZE, strides=1, padding='SAME', activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=x)


def build_autoencoder():
    encoder = build_encoder()
    decoder = build_decoder()
    autoencoder = tf.keras.models.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )

    return encoder, decoder, autoencoder
