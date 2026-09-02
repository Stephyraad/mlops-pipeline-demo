import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score, 
    precision_score
)

def calculate_metrics(target:pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        'f1 Score': f1_score(target, predictions),
        'Recall Score': recall_score(target, predictions),
        'Accuracy Score': accuracy_score(target, predictions),
        'Precision Score':precision_score(target, predictions)
    }


def print_metrics(model_type: str, metrics: dict[str, float]):
    print("\nModel Performance")
    print("-" * 30)
    print(f"{'Model:':<15} {model_type}")
    print()
    for name, value in metrics.items():
        print(f"{name:<20} {value:.4f}")
    print()
