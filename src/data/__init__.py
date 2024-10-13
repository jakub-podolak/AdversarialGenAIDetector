from .datasets import (
    BaseDataset,
    TwitterBotDataset,
    TwiBot22Dataset,
    FollowUpQGDataset,
    ResearchAbstractsDataset,
    ChatGPTArticleDataset,
    HC3Dataset,
    XsumLlama
)
from .datasets import (
    TWITTER_BOT_DATASET_PATH,
    FOLLOWUP_QG_DATASET_PATH,
    RESEARCH_ABSTRACTS_PATH,
    CHATGPT_ARTICLE_PATH,
    HC3_DATASET_PATH,
    TWIBOT22_DATASET_PATH,
    XSUM_LLAMA_DATASET_PATH,
    XSUM_VICUNA_DATASET_PATH,
)


DATASETS: dict[str, callable] = {
    "TwitterBotDataset": (TwitterBotDataset, TWITTER_BOT_DATASET_PATH),
    "FollowUpQGDataset": (FollowUpQGDataset, FOLLOWUP_QG_DATASET_PATH),
    "ResearchAbstracts": (ResearchAbstractsDataset, RESEARCH_ABSTRACTS_PATH),
    "ChatGPTArticles": (ChatGPTArticleDataset, CHATGPT_ARTICLE_PATH),
    "HC3Dataset": (HC3Dataset, HC3_DATASET_PATH),
    "Twibot22Dataset": (TwiBot22Dataset, TWIBOT22_DATASET_PATH),
    "XsumLlama": (XsumLlama, XSUM_LLAMA_DATASET_PATH),
    "XsumVicuna": (XsumLlama, XSUM_VICUNA_DATASET_PATH),   
}


def get_dataset(
    dataset_name: str,
    tokenize: callable = None,
    data_augmentations: callable = None,
    is_test: bool = False,
) -> BaseDataset:
    dataset, dataset_path = DATASETS[dataset_name]
    return dataset(
        data_path=dataset_path,
        tokenize=tokenize,
        data_augmentations=data_augmentations,
        is_test=is_test,
    )
