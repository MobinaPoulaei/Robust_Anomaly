import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class MVTecDataset(Dataset):
    def __init__(self, root_dir, class_name, mode='train', transform=None, mask_transform=None):
        """
        Custom dataset for MVTec-AD.

        Parameters:
        ----------
        root_dir : str
            Root directory where the MVTec-AD dataset is stored.
        class_name : str
            The name of the class (e.g., 'bottle').
        mode : str
            'train' or 'test'. Determines which part of the dataset to load.
        transform : torchvision.transforms.Compose, optional
            Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.transform = transform
        self.mask_transform = mask_transform


        self.images, self.labels, self.masks = self._prepare_data()

    def _prepare_data(self):
        images = []
        labels = []
        masks = []
        image_paths = []

        if self.mode == 'train':
            train_dir = os.path.join(self.root_dir, self.class_name, 'train', 'good')
            for img_file in os.listdir(train_dir):
                images.append(os.path.join(train_dir, img_file))
                labels.append(0)
                masks.append(None)

        elif self.mode == 'test':
            test_dir = os.path.join(self.root_dir, self.class_name, 'test')
            for sub_dir in os.listdir(test_dir):
                sub_dir_path = os.path.join(test_dir, sub_dir)
                if sub_dir == 'good':
                    label = 0
                    for img_file in os.listdir(sub_dir_path):
                        images.append(os.path.join(sub_dir_path, img_file))
                        labels.append(label)
                        masks.append(None)
                else:
                    label = 1
                    gt_dir = os.path.join(self.root_dir, self.class_name, 'ground_truth', sub_dir)
                    for img_file in os.listdir(sub_dir_path):
                        images.append(os.path.join(sub_dir_path, img_file))
                        labels.append(label)
                        mask_path = os.path.join(gt_dir, img_file.replace('.png', '_mask.png'))
                        masks.append(mask_path if os.path.exists(mask_path) else None)

        else:
            raise ValueError("Mode must be 'train' or 'test'.")

        return images, labels, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        mask = torch.zeros((1, 224, 224))
        if mask_path:
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)

        return image, label#, mask, image_path