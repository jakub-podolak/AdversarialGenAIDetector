import torch.nn as nn

from argparse import ArgumentParser
from models import MODEL_DICT


def parse_args():
    parser = ArgumentParser()

    # General
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for the model",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["TwitterBotDataset", "Twibot22Dataset"],
        default="Twibot22Dataset",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--with_data_augmentations",
        action="store_true",
        help="Flag to indicate data augmentations",
    )

    # Model & Parameters
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_DICT.keys(),
        default="RobertaLarge",
        help="Name of the model to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader",
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()
