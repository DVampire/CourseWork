import os

from PIL import Image
from torch.utils.data import Dataset

from src.logger import logger
from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = assemble_project_path(image_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(os.listdir(self.image_dir))
        }

        logger.info(f'Found {len(self.class_to_idx)} classes in the dataset.')
        for cls_name, idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.image_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
