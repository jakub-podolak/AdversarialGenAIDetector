import os
import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaForSequenceClassification, PreTrainedModel, PretrainedConfig


class RobertaLarge(PreTrainedModel):
    def __init__(self, num_classes: int, device: torch.device = torch.device('cuda'), checkpoint: str = "FacebookAI/roberta-large") -> None:
        config = PretrainedConfig()
        config.num_labels = num_classes
        config.from_pretrained(checkpoint)
        super().__init__(config)

        self.checkpoint = checkpoint
        self.model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes).to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(checkpoint)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask, labels=labels)
        return output

    def tokenize(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding="max_length", truncation=True, max_length=256)
        return {k: v.squeeze(0) for k, v in x.items()}

    # def from_pretrained(self, checkpoint: str):
    #     self.model.load_state_dict(torch.load(checkpoint))
    #     return self
    
    # def from_pretrained(self, checkpoint: str):
    #     self.model.from_pretrained(checkpoint)
    #     return self
