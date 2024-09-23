from .datasets import BaseDataset, TwitterBotDataset, FollowUpQGDataset
from .datasets import TWITTER_BOT_DATASET_PATH, FOLLOWUP_QG_DATASET_PATH


DATASETS: dict[str, callable] = {
    "TwitterBotDataset": (TwitterBotDataset, TWITTER_BOT_DATASET_PATH),
    "FollowUpQGDataset": (FollowUpQGDataset, FOLLOWUP_QG_DATASET_PATH),
}


def get_dataset(
    dataset_name: str,
    data_transforms: callable = None,
    data_augmentations: callable = None,
    is_test: bool = False,
) -> BaseDataset:
    dataset, dataset_path = DATASETS[dataset_name]
    return dataset[dataset_name](
        data_path=dataset_path,
        data_transforms=data_transforms,
        data_augmentations=data_augmentations,
        is_test=is_test,
    )
