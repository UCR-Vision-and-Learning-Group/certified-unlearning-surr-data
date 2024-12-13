from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset
import numpy as np

from typing import Tuple


def get_dataloaders(datasets,
                    batch_size=64,
                    shuffle=True):
    if isinstance(datasets, Dataset):
        return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloaders: list[DataLoader] = []
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, Dataset):
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
        return dataloaders


def get_train_test_datasets(idx: str, transform, target_transform=None) -> Tuple[Dataset, Dataset]:
    train_dataset, test_dataset = None, None

    if idx == 'mnist':
        train_dataset = MNIST('./data', train=True, transform=transform,
                              target_transform=target_transform, download=True)
        test_dataset = MNIST('./data', train=False, transform=transform,
                             target_transform=target_transform, download=True)
    try:
        assert (test_dataset is not None and train_dataset is not None), 'the given dataset id is not recognized'
    except AssertionError as error:
        print(error)

    return train_dataset, test_dataset


def get_retain_forget_datasets(dataset, forget):
    retain_dataset, forget_dataset = None, None
    if isinstance(float, forget):
        # selective unlearning
        forget_size = int(len(dataset) * forget)
        retain_size = len(dataset) - forget_size
        retain_dataset, forget_dataset = random_split(dataset, [retain_size, forget_size])
    elif isinstance(int, forget):
        # class unlearning
        retain_dataset, forget_dataset = [], []
        for idx, (_, label) in enumerate(dataset):
            if label != forget:
                retain_dataset.append(idx)
            else:
                forget_dataset.append(idx)
        retain_dataset = Subset(dataset, retain_dataset)
        forget_dataset = Subset(dataset, forget_dataset)

    return retain_dataset, forget_dataset


def get_class_ratios(dataset, num_class):
    ratios = np.zeros(num_class, dtype=int)
    for _, label in dataset:
        ratios[label] += 1
    return ratios / len(dataset)


def _partite_by_class(dataset, num_class):
    idxs = [[] for _ in range(num_class)]
    for idx, (_, label) in enumerate(dataset):
        idxs[label].append(idx)
    return [Subset(dataset, idx) for idx in idxs]


def get_exact_surr_datasets(dataset, num_class, target_size=None, target_ratios=None, surr_dataset=None):
    # TODO: errors might be explained later
    if surr_dataset is None and (target_ratios is not None and target_size is not None):
        # partite all classes
        class_partitions = _partite_by_class(dataset, num_class)

        # find target sizes for each class
        target_sizes = (target_ratios * target_size).astype(int)
        remainder = target_size - np.sum(target_sizes)
        add_to = np.random.choice(num_class, remainder)
        for add_to_idx in add_to:
            target_sizes[add_to_idx] += 1
        assert np.sum(target_sizes) == target_size, 'target size could not achieved'

        # randomly select specified number of samples from each class
        # TODO: for the second dataset the same thing can be applied as the first one
        first_class_partitions, second_class_partitions = [], []
        for class_idx, target_size_by_class in enumerate(target_sizes):
            first_idx = np.random.choice(len(class_partitions[class_idx]), target_size_by_class, replace=False)
            second_idx = np.delete(np.arange(len(class_partitions[class_idx])), first_idx)
            first_class_partitions.append(Subset(class_partitions[class_idx], first_idx))
            second_class_partitions.append(Subset(class_partitions[class_idx], second_idx))
        return ConcatDataset(first_class_partitions), ConcatDataset(second_class_partitions)
    elif surr_dataset is not None:
        return dataset, surr_dataset
