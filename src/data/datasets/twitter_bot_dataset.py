import pandas as pd

from .base_dataset import BaseDataset


TWITTER_BOT_DATASET_PATH = "raw_data/bot_detection_data.csv"


class TwitterBotDataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        df = pd.read_csv(data_path)
        return df["Tweet"].tolist(), df["Bot Label"].tolist()
