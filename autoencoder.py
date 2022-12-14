from models.autoencoder import build_autoencoder, build_autoencoder_callbacks
from loaders.nga_dataset import nga_dataset_generator, load_nga_dataset

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import argparse

DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAINING_STEPS_PER_EPOCH = 450
DEFAULT_TRAINING_EPOCHS = 2
SEED = 13


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ArtML',
        description='Self organizing maps on National Gallery of Art dataset',
        add_help=True)

    parser.add_argument('dataset', help="NGA dataset folder")
    parser.add_argument('--batch', '--b', nargs="?", type=int, help="autoencoder training batch size",
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', '--e', nargs="?", type=int, help="autoencoder training epochs",
                        default=DEFAULT_TRAINING_EPOCHS)
    parser.add_argument('--plots', '--p', action="store_true",
                        help="generate autoencoder plots on graphs folder", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    encoder, decoder, autoencoder = build_autoencoder()
    dataset = load_nga_dataset(args.dataset, mode='training', split=1.0)
    train_x, test_x, train_y, test_y = train_test_split(dataset, dataset, test_size=0.3, random_state=SEED)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.3, random_state=SEED)

    encoder.summary()
    decoder.summary()

    history = autoencoder.fit(train_x, train_y, epochs=args.epochs, callbacks=build_autoencoder_callbacks(args.dataset,
                                                                                                          args.plots),
                              validation_data=(valid_x, valid_y), verbose=True)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.gca()
    plt.grid(True)
    plt.savefig('graphs/autoencoder-training.png')

    results = autoencoder.evaluate(test_x, test_y)
    print("Evaluation on test set:", results)
