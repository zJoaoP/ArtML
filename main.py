

from loaders import download_nga_dataset
from dataset import SingleClassDataset
from torchvision import transforms
from torch.utils import data

DATASET_FOLDER = 'nga_dataset'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

if __name__ == '__main__':
    download_nga_dataset(DATASET_FOLDER)
    nga_dataset = SingleClassDataset(DATASET_FOLDER, transforms.Compose([transforms.Resize(size=IMAGE_SIZE),
                                                                         transforms.ToTensor()]))

    train_loader = data.DataLoader(nga_dataset, batch_size=BATCH_SIZE, drop_last=True)

    for idx, img in enumerate(train_loader):
        print(img.shape)
        break
