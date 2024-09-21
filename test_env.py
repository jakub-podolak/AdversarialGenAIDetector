import torch
from data.datasets import DATASETS

print("Is cuda available:", torch.cuda.is_available())
print("Registered datasets:")
for dataset in DATASETS.keys():
    print(dataset)
    dataset_df = DATASETS[dataset]().dataframe
    print(len(dataset_df), "Rows, out of which positives:",\
          len(dataset_df.query("label == 1")))