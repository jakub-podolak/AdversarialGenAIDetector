import pandas as pd
from dataclasses import dataclass

REQUIRED_COLUMNS: list[str] = ["text", "label"]

@dataclass
class Dataset:
    # Dataframe with columns ['text', 'label'] where
    # label = 1 means that text is AI generated
    dataframe: pd.DataFrame

    def __init__(self, raw_data: pd.DataFrame):
        for column in REQUIRED_COLUMNS:
            assert column in raw_data.columns

        self.dataframe = raw_data


# I recommend to create functions like this one for new datasets
def create_sample_dataset() -> Dataset:
    df = pd.DataFrame([
        {"text": "Man i just love cooking spathetti with my friends xd",
         "label": 0},
        {"text": "Cooking spaghetti with friends is more than just making a"
                  "meal—it's a shared experience of joy and connection.",
         "label": 1}
    ])
    return Dataset(df)


# Register all the datasets in __init__.py, so we can just use
# `dataset = DATASETS["sample_dataset"]()` in code
