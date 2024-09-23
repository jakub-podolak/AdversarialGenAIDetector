from .base_dataset import BaseDataset


FOLLOWUP_QG_DATASET_PATH = "raw_data/followup_questions.json"


class FollowUpQGDataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)

    def load_data_and_labels(self, data_path: str) -> tuple[list, list]:
        # TODO
        pass
