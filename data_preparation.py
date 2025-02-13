import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random


def download_mnist(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])  # transforms.Normalize((0.5,), (0.5,))
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def create_imbalanced_dataset(dataset, reduced_classes, reduction_factor=0.1):
    indices = {label: [] for label in range(10)}
    for idx, (image, label) in enumerate(dataset):
        indices[label].append(idx)

    imbalanced_indices = []
    for label in range(10):
        if label in reduced_classes:
            imbalanced_indices.extend(indices[label][:int(len(indices[label]) * reduction_factor)])
        else:
            imbalanced_indices.extend(indices[label])

    return Subset(dataset, imbalanced_indices)


def create_class_removed_dataset(dataset, removed_classes):
    indices = []
    for idx, (image, label) in enumerate(dataset):
        if label not in removed_classes:
            indices.append(idx)
    return Subset(dataset, indices)


def create_datasets(data_dir):
    train_dataset, test_dataset = download_mnist(data_dir)

    original_train_dataset = train_dataset
    original_test_dataset = test_dataset

    rare_train_dataset = create_class_removed_dataset(train_dataset, removed_classes=[0, 1, 2, 3, 4, 6, 7, 8, 9])
    rare_test_dataset = create_class_removed_dataset(test_dataset, removed_classes=[0, 1, 2, 3, 4, 6, 7, 8, 9])

    common_train_dataset = create_class_removed_dataset(train_dataset, removed_classes=[5])
    common_test_dataset = create_class_removed_dataset(test_dataset, removed_classes=[5])

    return (original_train_dataset, original_test_dataset,
            rare_train_dataset, rare_test_dataset,
            common_train_dataset, common_test_dataset)


if __name__ == "__main__":
    data_dir = "./mnist_data"
    create_datasets(data_dir)
  
