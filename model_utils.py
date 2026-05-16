"""
Shared utilities for the LSTM sepsis prediction pipeline.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
)


# qSOFA and SOFA constants

QSOFA_RESP_THRESHOLD: int = 22    # Respiratory rate >= 22/min: respiratory risk signal.
QSOFA_SBP_THRESHOLD:  int = 100   # Systolic blood pressure <= 100 mmHg: hypotension.

# SOFA - Bilirubin (liver) thresholds (mg/dL)
SOFA_LIVER_THRESHOLDS: list[float] = [0, 1.2, 2.0, 6.0, 12.0, 1000.0]
SOFA_LIVER_SCORES:     list[int]   = [0, 1,   2,   3,   4,    4]

# SOFA - Platelet thresholds (x10^3/uL)
SOFA_PLATELET_THRESHOLDS: list[float] = [0, 20, 50, 100, 150, 10_000]
SOFA_PLATELET_SCORES:     list[int]   = [4, 3,  2,  1,   0,   0]

# SOFA - MAP thresholds (mmHg)
SOFA_MAP_THRESHOLDS: list[float] = [0, 70, 1000]
SOFA_MAP_SCORES:     list[int]   = [1, 0,  0]

# Time window for SOFA worst-case aggregation (hours).
SOFA_WINDOW_HOURS: int = 24


# 1. Model architecture

def create_bilstm(
    n_units:    int,
    dropout:    float,
    seq_len:    int,
    n_features: int,
    lr:         float = 1e-3,
) -> tf.keras.Model:

    model = Sequential([
        Bidirectional(
            LSTM(n_units, return_sequences=True),
            input_shape=(seq_len, n_features),
        ),
        Dropout(dropout),
        BatchNormalization(),

        Bidirectional(
            LSTM(max(n_units // 2, 8), return_sequences=False)
        ),
        Dropout(dropout),
        BatchNormalization(),

        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(curve="ROC", name="auroc"),
            tf.keras.metrics.AUC(curve="PR",  name="auprc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


# 2. Standard training callbacks

def get_callbacks(
    checkpoint_path: str   = "best_model.keras",
    monitor:         str   = "val_auprc",
) -> list:
 
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, reduce_lr, checkpoint]


# 3. Model evaluation


def eval_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold":   round(float(threshold), 4),
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "youden_j":    sensitivity + specificity - 1,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


def find_best_threshold(
    y_true:              np.ndarray,
    y_prob:              np.ndarray,
    min_sensitivity:     float = 0.80,
    threshold_range:     tuple[float, float] = (0.05, 0.95),
    step:                float = 0.01,
) -> dict[str, float]:
 
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    rows = [eval_at_threshold(y_true, y_prob, th) for th in thresholds]

    candidates = [r for r in rows if r["sensitivity"] >= min_sensitivity]

    if candidates:
        best = max(candidates, key=lambda r: (r["specificity"], r["youden_j"]))
    else:
        print(f"No threshold reached sensitivity >= {min_sensitivity}. "
              "Falling back to Youden's J.")
        best = max(rows, key=lambda r: r["youden_j"])

    return best


def full_evaluation(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float,
    label:     str = "TEST",
) -> None:

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    m     = eval_at_threshold(y_true, y_prob, threshold)

    print(f"\n{'='*55}")
    print(f"  {label} - threshold = {threshold:.2f}")
    print(f"{'='*55}")
    print(f"  AUROC        : {auroc:.4f}")
    print(f"  AUPRC        : {auprc:.4f}")
    print(f"  Accuracy     : {m['accuracy']:.4f}")
    print(f"  Sensitivity  : {m['sensitivity']:.4f}")
    print(f"  Specificity  : {m['specificity']:.4f}")
    print(f"  Precision    : {m['precision']:.4f}")
    print(f"  Recall       : {m['recall']:.4f}")
    print(f"  F1-score     : {m['f1']:.4f}")
    print(f"  Youden's J   : {m['youden_j']:.4f}")
    print(f"  Confusion    : TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print(f"{'='*55}\n")
