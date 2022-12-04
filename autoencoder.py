from models.autoencoder import build_autoencoder, build_autoencoder_callbacks
from loaders.nga_dataset import nga_dataset_generator, load_nga_dataset

import matplotlib.pyplot as plt
import pandas as pd
import argparse

DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAINING_STEPS_PER_EPOCH = 450
DEFAULT_TRAINING_EPOCHS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ArtML',
        description='Self organizing maps on National Gallery of Art dataset',
        add_help=True)

    parser.add_argument('dataset', help="NGA dataset folder")
    parser.add_argument('--batch', '--b', nargs="?", help="autoencoder training batch size", default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', '--e', nargs="?", help="autoencoder training epochs",
                        default=DEFAULT_TRAINING_EPOCHS)

    parser.add_argument('--steps', '--s', nargs="?", help="autoencoder training steps per epoch",
                        default=DEFAULT_TRAINING_STEPS_PER_EPOCH)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # download_nga_dataset(args.dataset)

    encoder, decoder, autoencoder = build_autoencoder()
    validation_images = load_nga_dataset(args.dataset, mode='validation', split=0.15)
    encoder.summary()
    decoder.summary()

    history = autoencoder.fit(nga_dataset_generator(args.dataset, args.batch, mode='training', split=0.85),
                              epochs=args.epochs, steps_per_epoch=args.steps,
                              callbacks=build_autoencoder_callbacks(args.dataset),
                              validation_data=(validation_images, validation_images), verbose=True)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.gca()
    plt.grid(True)
    plt.show()
