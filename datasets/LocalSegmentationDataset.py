from torch.utils.data import Dataset
import glob
import os
import random
from PIL import Image
from utils.image_utils import load_image
import numpy as np

class LocalSegmentationDataset(Dataset):

    def __init__(self, root_dir='./', image_dir='images/*', label_dir='labels/*', split=False, train=True, min_size=None, ignore_zero=False, transform=None):
        
        image_list = glob.glob(os.path.join(root_dir, image_dir))
        image_list.sort()

        label_list = glob.glob(os.path.join(root_dir, label_dir))
        label_list.sort()

        assert len(image_list) == len(label_list), f'Length image_list ({len(image_list)}) does not equal length label_list ({len(label_list)})'

        if split:
            n_samples = len(image_list)
            image_list = image_list[0: round(n_samples*0.8)] if train else image_list[round(n_samples*0.8):]
            label_list = label_list[0: round(n_samples*0.8)] if train else label_list[round(n_samples*0.8):]

        self.image_list = image_list
        self.label_list = label_list
        self.size = len(image_list)
        self.min_size = min_size
        self.ignore_zero = ignore_zero
        self.transform = transform

        print(f'SegmentationDataset created: found {self.size} images')

    def __len__(self):
        return self.min_size or self.size

    def get_index(self, index):
        if self.min_size:
            index = random.randint(0, self.size - 1) if self.size > 1 else 0
        return index

    def get_image(self, index):
        image_path = self.image_list[index]
        image = load_image(image_path)
        return image

    def get_label(self, index):
        label_path = self.label_list[index]
        label = np.array(Image.open(label_path))

        if self.ignore_zero:
            label = label - 1

        return label

    def __getitem__(self, index):
        index = self.get_index(index)

        image = np.array(self.get_image(index))
        label = self.get_label(index)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        sample = {
            "image": image,
            "label": label,
        }

        return sample