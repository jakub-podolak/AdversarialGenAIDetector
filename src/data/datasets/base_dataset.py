from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_transforms: callable = None,
        data_augmentations: callable = None,
        is_test: bool = False,
    ):
        self.data, self.labels = self.load_data_and_labels(data_path)
        self.data_transforms = data_transforms
        self.data_augmentations = data_augmentations
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        sample, label = self.data[idx], self.labels[idx]
        if self.is_test and self.data_augmentations:
            sample = self.data_augmentations(sample)
        if self.data_transforms:
            sample = self.data_transforms(sample)
        return sample, label

    @abstractmethod
    def load_data_and_labels(self, data_path, *args, **kwargs) -> tuple[list, list]:
        pass
