import numpy as np
from src.evaluation import evaluate_model


def test_evaluate_model_returns_all_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.2, 0.8, 0.4, 0.3])

    metrics = evaluate_model(y_true, y_pred, y_proba)

    assert "roc_auc" in metrics
    assert "f1" in metrics
