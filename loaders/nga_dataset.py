import multiprocessing
import pandas as pd
import threading
import requests
import os

from queue import Queue

MAX_WIDTH = MAX_HEIGHT = 256


def load_nga_dataset(location="https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data"
                              "/published_images.csv"):
    return pd.read_csv(location)


def is_dataset_downloaded(folder='nga_dataset', dataset=None):
    if dataset is None:
        dataset = load_nga_dataset("nga_dataset.csv")

    items_on_dir = os.listdir('nga_dataset')
    os.listdir('nga_dataset')
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
            uuid, width, height = q.get()

            image_url = "https://api.nga.gov/iiif/{}/full/!{},{}/0/default.jpg".format(uuid, width, height)
            image_path = get_destination_filepath(uuid)
            if os.path.exists(image_path):
                continue

            img_data = requests.get(image_url).content
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
    return dataset
