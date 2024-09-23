from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="TwitterBotDataset",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the dataloader",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    parser.add_argument(
        "--is_test",
        type=bool,
        default=False,
        help="Flag to indicate test mode",
    )
    return parser.parse_args()
