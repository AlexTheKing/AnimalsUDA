import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler



class AnimalsSupervisedDataset(Dataset):
    def __init__(self, folder, labels, transforms=None):
        self.folder = folder
        self.labels = labels
        self.transforms = transforms
        self.filenames = sorted(
            file for file in os.listdir(self.folder)
            if file != '.DS_Store'
        )

    def __len__(self):
        return len(self.filenames)

    def get_label_for_target(self, target):
        return self.labels.columns[np.argmax(target)]

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.folder, img_filename)
        img_id = img_filename.split('.')[0]  # remove .jpg extension
        target = self.labels.loc[img_id].to_numpy()

        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, target


class AnimalsUnsupervisedDataset(Dataset):
    def __init__(self, folder, resize_transforms, augment_transforms=None):
        self.folder = folder
        self.resize_transforms = resize_transforms
        self.augment_transforms = augment_transforms
        self.filenames = sorted(
            file for file in os.listdir(self.folder)
            if file != '.DS_Store'
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.folder, img_filename)

        original_image = np.array(Image.open(img_path).convert('RGB'))
        original_image_resized = self.resize_transforms(image=original_image)['image']

        if self.augment_transforms:
            augmented_image = self.augment_transforms(image=original_image)['image']
        else:
            augmented_image = np.array([])

        return original_image_resized, augmented_image


class WeightedSubsetRandomSampler(Sampler):
    def __init__(self, indices, weights, replacement=True):
        self.weights = weights
        self.indices = indices
        self.num_samples = len(indices)
        self.replacement = replacement

    def __iter__(self):
        for i in torch.multinomial(self.weights, self.num_samples, self.replacement):
            yield self.indices[i]

    def __len__(self):
        return self.num_samples


class AnimalsUDAData:
    def __init__(
            self,
            train_features='train_features.csv',
            train_labels='train_labels.csv',
            train_folder='train_features',
            test_folder='test_features',
            resize=(224, 224),
            supervised_batch_size=16,  # UDA Paper: 64 sup & 448 unsup
            unsupervised_ratio=7  # in paper 7
    ):
        self.train_features = pd.read_csv(train_features)
        self.train_labels = pd.read_csv(train_labels, index_col='id')
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.resize = resize
        self.supervised_batch_size = supervised_batch_size
        self.unsupervised_batch_size = supervised_batch_size * unsupervised_ratio
        print(f'Number of examples in {train_folder} is {len(os.listdir(train_folder))}')
        print(f'Number of examples in {test_folder} is {len(os.listdir(test_folder))}')
        self.train_indices, self.validation_indices = self.get_train_validation_indices()

    def get_train_validation_indices(self):
        validation_sites = self.train_features['site'].value_counts()[-80:].index.to_list()  # around 2k images
        print(f'Got around {len(validation_sites)} sites in validation')

        validation_indices = self.train_features[self.train_features['site'].isin(validation_sites)].index.to_list()
        train_indices = list(set(self.train_features.index.to_list()) - set(validation_indices))

        print(f'Got {len(train_indices)} train examples and {len(validation_indices)} validation examples')

        return train_indices, validation_indices

    def sample_datasets(self):
        resize_and_normalize = A.Compose([
            A.Resize(*self.resize),
            A.Normalize(),
            ToTensorV2()
        ])
        augment = A.Compose([
            A.Resize(*self.resize),
            *RandAugment(transformations_count=8, magnitude=10, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
        supervised_dataset = AnimalsSupervisedDataset(
            self.train_folder, labels=self.train_labels,
            transforms=augment
        )
        unsupervised_dataset = AnimalsUnsupervisedDataset(
            self.test_folder,
            resize_transforms=resize_and_normalize,
            augment_transforms=augment
        )
        validation_dataset = AnimalsSupervisedDataset(
            self.train_folder, labels=self.train_labels,
            transforms=resize_and_normalize
        )
        return supervised_dataset, unsupervised_dataset, validation_dataset

    def sample_loaders(self):
        supervised_dataset, unsupervised_dataset, validation_dataset = self.sample_datasets()
        train_supervised_loader = DataLoader(
            supervised_dataset, batch_size=self.supervised_batch_size,
            sampler=WeightedSubsetRandomSampler(  # class balancing
                self.train_indices, torch.full(size=(len(self.train_indices),), fill_value=0.5)
            ),
            pin_memory=True
        )
        train_unsupervised_loader = DataLoader(
            unsupervised_dataset, batch_size=self.unsupervised_batch_size, shuffle=True,
            pin_memory=True
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.supervised_batch_size,
            sampler=SubsetRandomSampler(self.validation_indices),
            pin_memory=True
        )
        return train_supervised_loader, train_unsupervised_loader, validation_loader

    def get_test_dataset(self):
        return AnimalsUnsupervisedDataset(self.test_folder, resize_transforms=A.Compose([
            A.Resize(*self.resize),
            A.Normalize(),
            ToTensorV2()
        ]))
