import torch

from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

from logger import logger
from config import parse_args
from data import get_dataset
from utils.metrics import compute_metrics
from models import get_model
from augmentation import create_augmentation_pipeline


def create_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        no_cuda=args.device != torch.device("cuda"),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
    )


if __name__ == "__main__":
    args = parse_args()
    training_args = create_training_args(args)

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Train dataset: {args.dataset}")
    
    # creating augmentation pipeline
    augmentation_pipeline = create_augmentation_pipeline()

    # Dataset
    train_dataset = get_dataset(args.dataset, is_test=False)
    val_dataset = get_dataset(args.dataset, is_test=args.is_test)

    train_dataset.set_data_transforms(lambda text: augmentation_pipeline(text))
    val_dataset.set_data_transforms(lambda text: augmentation_pipeline(text))

    # Model
    model = get_model(args.model, num_classes=train_dataset.num_classes, device=args.device).to(args.device)
    train_dataset.set_data_transforms(model.tokenize)
    val_dataset.set_data_transforms(model.tokenize)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
    )

    trainer.train()
