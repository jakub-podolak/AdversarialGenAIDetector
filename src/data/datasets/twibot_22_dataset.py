import os
import torch
import ijson
import pandas as pd

from collections import defaultdict
from .base_dataset import BaseDataset


TWIBOT22_DATASET_PATH = "raw_data/twibot22"


class TwiBot22Dataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def load_data_and_labels(self, data_path: str) -> tuple[torch.tensor, torch.tensor]:
        dataset_file = "tweet_0.json" if self.is_test else "tweet_1.json"
        print("Using dataset file: ", dataset_file, "since is_test is", self.is_test)
        labels = self.load_all_labels(os.path.join(data_path, "label.csv"))
        data, labels = self._load_json_data_with_labels(os.path.join(data_path, dataset_file), labels)
        assert len(data) == len(labels)
        return data, labels

    def _load_json_data_with_labels(self, data_path: str, labels: pd.DataFrame) -> list:
        label_counts = defaultdict(int)
        num_class_samples = 500 if self.is_test else 5000
        with open(data_path, "r") as f:
            objects = ijson.items(f, "item")
            texts = []
            filtered_labels = []
            for i, obj in enumerate(objects):
                if label_counts[0] == num_class_samples and label_counts[1] == num_class_samples:
                    break
                author_id = f"u{obj['author_id']}"
                label = int(labels["label"][labels["id"] == author_id].values[0])
                
                if label_counts[label] >= num_class_samples:
                    continue
                label_counts[label] += 1
                
                texts.append(obj["text"])
                filtered_labels.append(label)
        if self.data_augmentations:
            texts = [self.data_augmentations(text) for text in texts]
        if self.tokenize:
            texts = [self.tokenize(text) for text in texts]
        print("Loaded label counts: ", label_counts)
        return texts, filtered_labels

    def load_all_labels(self, labels_path: str) -> pd.DataFrame:
        # Load labels from a csv file
        labels_df = pd.read_csv(labels_path)
        labels_df["label"].replace({"bot": 1, "human": 0}, inplace=True)
        return labels_df
