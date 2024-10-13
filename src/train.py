import torch

from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

from logger import logger
from config import parse_args
from data import get_dataset
from utils.metrics import compute_metrics
from models import get_model
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from data.transformation.augmentation import create_augmentation_pipeline_proba


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
        save_total_limit=2,
    )


if __name__ == "__main__":
    args = parse_args()
    training_args = create_training_args(args)

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Train dataset: {args.dataset}")
    logger.info(f"Using device: {args.device}")

    # creating augmentation pipeline
    if args.with_data_augmentations:
        logger.info("Creating augmentation pipeline")
        augmentation_pipeline = create_augmentation_pipeline_proba()
    else:
        logger.info("Training without data augmentations")
        augmentation_pipeline = None

    # Model
    # model = get_model(args.model, num_classes=args.num_classes, device=args.device).to(
    #     args.device
    # )

    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=args.num_classes).to(args.device)
    
    # model = RobertaForSequenceClassification.from_pretrained("output_roberta_test/checkpoint-1", num_labels=args.num_classes).to(args.device)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    def tokenize(x):
        x = tokenizer(x, return_tensors='pt', padding="max_length", truncation=True, max_length=256)
        # return x
        return {k: v.squeeze(0) for k, v in x.items()}
    
    # Dataset
    train_dataset = get_dataset(
        args.dataset,
        is_test=False,
        tokenize=tokenize,
        data_augmentations=augmentation_pipeline,
    )
    val_dataset = get_dataset(
        args.dataset,
        is_test=True,
        tokenize=tokenize,
        data_augmentations=augmentation_pipeline,
    )
    
    logger.info("Datasets loaded")

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
