from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenize: callable = None,
        data_augmentations: callable = None,
        is_test: bool = True,
    ):
        self.is_test = is_test
        self.data_augmentations = data_augmentations
        self.tokenize = tokenize
        self.data, self.labels = self.load_data_and_labels(data_path)
        self.num_classes = len(set(self.labels))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        sample, label = self.data[idx], self.labels[idx]
        return {**sample, "label": label}

    @abstractmethod
    def load_data_and_labels(self, data_path, *args, **kwargs) -> tuple[list, list]:
        pass
