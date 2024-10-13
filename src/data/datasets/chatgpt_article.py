import pandas as pd

from .base_dataset import BaseDataset


CHATGPT_ARTICLE_PATH = "raw_data/chatgpt_article.csv"

# TEST DATASET
class ChatGPTArticleDataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        df = pd.read_csv(data_path)
        return df["article"].tolist(), df["class"].tolist()
