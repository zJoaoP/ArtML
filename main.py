

from loaders.nga_dataset import download_nga_dataset, nga_dataset_generator
from models.autoencoder import build_autoencoder, build_autoencoder_callbacks
import argparse

DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAINING_STEPS_PER_EPOCH = 150
DEFAULT_TRAINING_EPOCHS = 1000


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
    encoder.summary()
    decoder.summary()

    autoencoder.fit(nga_dataset_generator(args.dataset, args.batch),
                    epochs=args.epochs,
                    steps_per_epoch=args.steps,
                    callbacks=build_autoencoder_callbacks(args.dataset))

