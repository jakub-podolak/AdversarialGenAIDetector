import pandas as pd

from .base_dataset import BaseDataset

RESEARCH_ABSTRACTS_PATH = "raw_data/research-abstracts-labeled.csv"

# TEST DATASET
class ResearchAbstractsDataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        data = pd.read_csv(data_path)

        return data['text'].to_list(), data['label'].to_list()
