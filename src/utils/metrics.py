import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = logits.argmax(axis=-1)
    y_pred_score = F.softmax(torch.tensor(logits), dim=-1)[:, 1].detach().cpu().numpy()

    return {
        "auc": roc_auc_score(y_true, y_pred_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
