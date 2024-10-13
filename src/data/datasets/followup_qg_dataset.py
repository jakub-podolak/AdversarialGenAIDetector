import json
import pandas as pd

from .base_dataset import BaseDataset

FOLLOWUP_QG_DATASET_PATH = "/home/scur1745/AdversarialGenAIDetector/raw_data/followupqg500.csv"

# TEST DATASET
class FollowUpQGDataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        texts = []
        labels = []

        df = pd.read_csv(data_path)

        texts = df['follow-up'].to_list() + df['generated-followup'].to_list()
        labels = [0] * len(df) + [1] * len(df)

        return texts, labels