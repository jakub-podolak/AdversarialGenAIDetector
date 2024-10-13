import pandas as pd
import json

from .base_dataset import BaseDataset

XSUM_LLAMA_DATASET_PATH = "/home/scur1745/AdversarialGenAIDetector/RADAR/data/Xsum/paired_corpus_500_llama.json"
XSUM_VICUNA_DATASET_PATH = "/home/scur1745/AdversarialGenAIDetector/RADAR/data/Xsum/paired_corpus_500_vicuna.json"

class XsumLlama(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
    
    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:

        with open(data_path, 'r') as file:
            data = json.load(file)

            human_texts = [item['human-text'] for item in data['train']]
            ai_texts = [item['ai-text'] for item in data['train']]

        return list(human_texts) + list(ai_texts), ([0] * len(human_texts) + [1] * len(ai_texts))