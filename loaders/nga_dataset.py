import multiprocessing

import PIL
import pandas as pd
import numpy as np
import threading
import requests
import math
import os

from queue import Queue
from PIL import Image


UNAVAILABLE_IMAGE_MESSAGE = "No image resource with that identifier could be located."
CACHE_FOLDER = 'cache'
IMAGE_WIDTH = IMAGE_HEIGHT = 32
MAX_WIDTH = MAX_HEIGHT = 256


def load_nga_dataset(location="https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data"
                              "/published_images.csv"):
    return pd.read_csv(location)


def check_downloaded_images(folder='nga_dataset'):
    print("Running dataset integrity check..")
    has_error = False
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            Image.open(filepath)
        except PIL.UnidentifiedImageError:
            print("The file {} is corrupted and needs to be downloaded again".format(filepath))
            os.remove(filepath)
            has_error = True

    return not has_error


def is_dataset_downloaded(folder='nga_dataset', dataset=None):
    if dataset is None:
        dataset = load_nga_dataset("nga_dataset.csv")

    print("NGA dataset already downloaded..")
    items_on_dir = os.listdir(folder)
    return len(dataset) == len(items_on_dir)


def download_nga_dataset(folder='nga_dataset'):
    concurrent_runs = multiprocessing.cpu_count()

    dataset = load_nga_dataset("nga_dataset.csv")
    if is_dataset_downloaded(folder, dataset):
        return dataset

    files_downloaded_lock = threading.Lock()
    files_downloaded = 0
    file_counter = 0
    q = Queue()

    def get_destination_filepath(uuid):
        return "{}/{}.jpg".format(folder, uuid)

    def download_worker():
        nonlocal files_downloaded
        while True:
            if q.empty():
                break
            else:
                uuid, width, height = q.get()

                image_url = "https://api.nga.gov/iiif/{}/full/!{},{}/0/default.jpg".format(uuid, width, height)
                image_path = get_destination_filepath(uuid)
                if os.path.exists(image_path):
                    continue

                img_data = requests.get(image_url).content
                if img_data.decode('utf-8').find(UNAVAILABLE_IMAGE_MESSAGE) != -1:
                    print("Couldn't find {} resource. Skipping this file..".format(uuid))
                    continue

                with open(image_path, 'wb') as handler:
                    handler.write(img_data)

                with files_downloaded_lock:
                    files_downloaded += 1
                    if files_downloaded % 10 == 0:
                        print("{} of {} images downloaded".format(files_downloaded, file_counter))

                q.task_done()

    for (index, row) in dataset.iterrows():
        width, height = min(MAX_WIDTH, row['width']), min(MAX_HEIGHT, row['height'])
        if not os.path.exists(get_destination_filepath(row['uuid'])):
            q.put((row['uuid'], width, height))
            file_counter += 1

    print("Downloading NGA dataset..")
    for _ in range(concurrent_runs):
        t = threading.Thread(target=download_worker, daemon=True)
        t.start()

    q.join()

    print("NGA dataset downloaded successfully!!")
    check_downloaded_images(folder)
    return dataset


def build_cache_name(mode, split):
    return "{}/{}-{}.cache.npy".format(CACHE_FOLDER, mode, split)


def load_nga_dataset(folder="nga_dataset", mode='training', split=0.8):
    cache_path = build_cache_name(mode, split)
    if os.path.isfile(cache_path):
        return np.load(cache_path)

    filenames = os.listdir(folder)
    filenames = filenames[:math.floor(len(filenames) * split)] \
        if mode == 'training' \
        else filenames[:-1][:math.floor(len(filenames) * split)]

    print("loading {} {} images..".format(len(filenames), mode))

    images = np.empty(shape=(len(filenames), 32, 32, 3), dtype=np.float32)
    for i, filename in enumerate(filenames):
        try:
            raw_image = Image.open(os.path.join(folder, filename)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            images[i] = (np.array(raw_image) / 255.0).astype(np.float32)
        except PIL.UnidentifiedImageError:
            images[i] = np.zeros(shape=(32, 32, 3))

    np.save(cache_path, images)
    return images


def nga_dataset_generator(folder="nga_dataset", batch_size=8, mode='training', split=0.8):
    filenames = os.listdir(folder)
    filenames = filenames[:math.floor(len(filenames) * split)] \
        if mode == 'training' \
        else filenames[:-1][:math.floor(len(filenames) * split)]

    print('initializing generator for {} {} samples'.format(len(filenames), mode))
    loaded_images = dict()

    while True:
        for batch_id in range(len(filenames) // batch_size):
            batch_filenames = filenames[batch_id * batch_size:(batch_id + 1) * batch_size]
            current_batch = np.empty(shape=(batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32)
            for i, image_name in enumerate(batch_filenames):
                if image_name not in loaded_images:
                    filepath = os.path.join(folder, image_name)
                    try:
                        raw_image = Image.open(filepath).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                        loaded_images[image_name] = (np.array(raw_image) / 255.0).astype(np.float32)
                    except PIL.UnidentifiedImageError:
                        loaded_images[image_name] = np.ones(shape=(32, 32, 3))

                current_batch[i] = loaded_images[image_name]

            yield current_batch, current_batch
