import os

from models.autoencoder import build_autoencoder, CHECKPOINT_PATH
from loaders.nga_dataset import IMAGE_WIDTH, IMAGE_HEIGHT
from models.som import SOM
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import argparse


DEFAULT_WEIGHTS_LOCATION = 'checkpoints'
DEFAULT_STEPS_COUNT = 100000

DEFAULT_NODE_COUNT = 10
NODE_DIMENTION = 32

GRID_WIDTH = 5

SIGMA_DECAY_RATE = 0.01
INITIAL_SIGMA = 0.1
MIN_SIGMA = 1e-2

ETA_DECAY_RATE = 3e-5
INITIAL_ETA = 1e-3
MIN_ETA = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ArtML',
        description='Self organizing maps on National Gallery of Art dataset',
        add_help=True)

    parser.add_argument('dataset', help="NGA dataset folder")
    parser.add_argument('--weights', '--w', help="Autoencoder weights path", default=CHECKPOINT_PATH)
    parser.add_argument('--nodes', '--n', help="SOM node count", default=DEFAULT_NODE_COUNT)
    parser.add_argument('--steps', '--s', help="SOM training step count", default=DEFAULT_STEPS_COUNT)

    return parser.parse_args()


def load_images_from(dataset, size=-1):
    if size != -1:
        imgnames = os.listdir(dataset)[:size]
    else:
        imgnames = os.listdir(dataset)

    images = np.empty(shape=(len(imgnames), 32, 32, 3))

    for i, imgname in enumerate(imgnames):
        filepath = os.path.join(args.dataset, imgname)
        raw_image = Image.open(filepath).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        images[i] = (np.array(raw_image) / 255.0).astype(np.float32)

    return imgnames, images


def get_first_n_index_of(predictions: list, node: int, size=9):
    indexed_predictions = list(zip(range(len(predictions)), predictions))
    indexed_predictions = sorted(indexed_predictions, key=lambda p: p[1][1])
    return [i for i, n in indexed_predictions if n[0] == node][:size]


def plot_grid(images, width, height):
    _, axs = plt.subplots(width, height, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img)

    plt.show()


if __name__ == '__main__':
    args = parse_args()

    encoder, decoder, autoencoder = build_autoencoder()
    autoencoder.load_weights(args.weights).expect_partial()

    names, images = load_images_from(args.dataset)
    encodings = encoder.predict(images)

    som_model = SOM(node_count=args.nodes, dimensions=NODE_DIMENTION,
                    initial_sigma=INITIAL_SIGMA, sigma_decay_rate=SIGMA_DECAY_RATE,
                    min_sigma=MIN_SIGMA, initial_eta=INITIAL_ETA,
                    eta_decay_rate=ETA_DECAY_RATE, min_eta=MIN_ETA)

    som_model.init_from_dataset(encodings)
    som_model.fit(encodings, args.steps)

    predictions = list()
    for sample in encodings:
        prediction = som_model.predict(sample)
        distance = som_model.euclidean_distance(sample, som_model.nodes[prediction]).sum()
        predictions.append((prediction, distance))

    for node in range(DEFAULT_NODE_COUNT):
        indexes = get_first_n_index_of(predictions, node, size=GRID_WIDTH ** 2)
        node_images = [images[i] for i in indexes]
        plot_grid(node_images, GRID_WIDTH, GRID_WIDTH)
