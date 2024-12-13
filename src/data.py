from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, random_split, Subset

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


def get_exact_surr_datasets(dataset: Dataset):
    pass
