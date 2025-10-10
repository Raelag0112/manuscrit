"""
Evaluation metrics for organoid classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


class OrganoidMetrics:
    """Metrics tracker for organoid classification"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, probs=None):
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        if probs is not None:
            self.all_probs.extend(probs)
    
    def compute(self):
        return compute_metrics(
            np.array(self.all_labels),
            np.array(self.all_preds)
        )
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self):
        return classification_report(self.all_labels, self.all_preds)

