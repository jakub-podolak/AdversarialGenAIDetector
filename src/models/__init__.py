import torch
from .roberta_large import RobertaLarge


MODEL_DICT = {
    "RobertaLarge": RobertaLarge,
}


def get_model(model_name: str, num_classes: int, device: torch.device = torch.device('cuda'), checkpoint: str = None):
    return MODEL_DICT[model_name](num_classes=num_classes, device=device, checkpoint=checkpoint)
