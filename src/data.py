from torchvision.datasets import MNIST, USPS, CIFAR10
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
    elif idx == 'usps':
        train_dataset = USPS('./data', train=True, transform=transform,
                             target_transform=target_transform, download=True)
        test_dataset = USPS('./data', train=False, transform=transform,
                            target_transform=target_transform, download=True)
    elif idx == 'cifar10':
        train_dataset = CIFAR10('./data', train=True, transform=transform, 
                                target_transform=target_transform, download=True)
        test_dataset = CIFAR10('./data', train=False, transform=transform,
                               target_transform=target_transform, download=True)
    try:
        assert (test_dataset is not None and train_dataset is not None), 'the given dataset id is not recognized'
    except AssertionError as error:
        print(error)

    return train_dataset, test_dataset


def get_retain_forget_datasets(dataset, forget):
    retain_dataset, forget_dataset = None, None
    if isinstance(forget, float):
        # selective unlearning
        forget_size = int(len(dataset) * forget)
        retain_size = len(dataset) - forget_size
        retain_dataset, forget_dataset = random_split(dataset, [retain_size, forget_size])
    elif isinstance(forget, int):
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


def __check_max_reached(sizes, max_sizes):
    max_reached = []
    for idx, max_size in enumerate(max_sizes):
        if sizes[idx] >= max_size:
            max_reached.append(idx)
            sizes[idx] = max_size
    return sizes, np.asarray(max_reached)


def _get_sizes(size, ratios, max_sizes):
    num_class = len(ratios)
    sizes = (ratios * size).astype(int)

    sizes, max_reached = __check_max_reached(sizes, max_sizes)
    remainder = size - np.sum(sizes)
    for _ in range(remainder):
        sizes[np.random.choice(np.delete(np.arange(num_class), np.asarray(max_reached)))] += 1
        sizes, max_reached = __check_max_reached(sizes, max_sizes)
        if len(max_reached) == num_class:
            break

    assert np.sum(sizes) == size, 'size could not achieved'
    return sizes


def get_exact_surr_datasets(dataset,
                            target_size=None, target_ratios=None,
                            starget_size=None, starget_ratios=None,
                            surr_dataset=None):
    # TODO: errors might be explained later
    if surr_dataset is None and (target_ratios is not None and target_size is not None):
        num_class = len(target_ratios)
        # partite all classes
        class_partitions = _partite_by_class(dataset, num_class)

        # find target sizes for each class
        max_sizes = [len(partition) for partition in class_partitions]
        target_sizes = _get_sizes(target_size, target_ratios, max_sizes)
        starget_sizes = None
        if starget_size is not None and starget_ratios is not None:
            starget_sizes = _get_sizes(starget_size, starget_ratios, max_sizes)

        # randomly select specified number of samples from each class
        first_class_partitions, second_class_partitions = [], []
        for class_idx, target_size_by_class in enumerate(target_sizes):
            first_idx = np.random.choice(len(class_partitions[class_idx]), target_size_by_class, replace=False)
            if starget_sizes is not None:
                second_idx = np.random.choice(len(class_partitions[class_idx]), starget_sizes[class_idx], replace=False)
            else:
                second_idx = np.delete(np.arange(len(class_partitions[class_idx])), first_idx)
            first_class_partitions.append(Subset(class_partitions[class_idx], first_idx))
            second_class_partitions.append(Subset(class_partitions[class_idx], second_idx))
        return ConcatDataset(first_class_partitions), ConcatDataset(second_class_partitions)
    elif surr_dataset is not None:
        return dataset, surr_dataset
