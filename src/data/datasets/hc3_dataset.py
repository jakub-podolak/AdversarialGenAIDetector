import pandas as pd

from datasets import load_dataset

from .base_dataset import BaseDataset

HC3_DATASET_PATH = "raw_data/test_hc3_QA.jsonl"



# # TRAIN DATASET
# class HC3Dataset(BaseDataset):
#     def __init__(self, data_path: str, **kwargs):
#         super().__init__(data_path, **kwargs)
    
#     def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
#         ds = load_dataset("Hello-SimpleAI/HC3", "all")

#         df = ds.data['train'].to_pandas()

#         human_answers = df['human_answers'].apply(lambda x: x[0]).to_list()
#         chatgpt_generations = df['chatgpt_answers'].apply(lambda x: None if len(x) == 0 else x[0]).dropna().to_list()
#         # Take first reply
#         return human_answers + chatgpt_generations, [0] * len(human_answers) + [1] * len(chatgpt_generations)


class HC3Dataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        df = pd.read_json(data_path, lines=True)
        return df['text'], df['label']

