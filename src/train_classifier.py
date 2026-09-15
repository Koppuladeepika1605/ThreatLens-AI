"""
ThreatLens AI - Binary Threat Classifier (HistGradientBoostingClassifier)
========================================================================

Role:
First-line detection model determining whether incoming network telemetry
represents Normal traffic (0) or Suspicious / Malicious Attack traffic (1).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src import config
from src.explainability import ModelExplainer

# Global cached model and preprocessor
_SUSPICIOUS_CLASSIFIER = None
_PREPROCESSOR = None
_FEATURE_NAMES = None


def get_preprocessor():
    """Load or return cached preprocessor pipeline."""
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {config.PREPROCESSOR_PATH}")
        _PREPROCESSOR = joblib.load(config.PREPROCESSOR_PATH)
    return _PREPROCESSOR


def get_suspicious_classifier():
    """Load or return cached suspicious classifier model."""
    global _SUSPICIOUS_CLASSIFIER
    if _SUSPICIOUS_CLASSIFIER is None:
        if not config.SUSPICIOUS_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Suspicious classifier not found at {config.SUSPICIOUS_CLASSIFIER_PATH}")
        _SUSPICIOUS_CLASSIFIER = joblib.load(config.SUSPICIOUS_CLASSIFIER_PATH)
    return _SUSPICIOUS_CLASSIFIER


def get_feature_names() -> List[str]:
    """Load or return cached feature names."""
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        if not config.FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError(f"Feature names not found at {config.FEATURE_NAMES_PATH}")
        with open(config.FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            _FEATURE_NAMES = json.load(f)
    return _FEATURE_NAMES


def train_binary_classifier(
    X_train_transformed: np.ndarray,
    y_train_binary: np.ndarray,
    random_state: int = config.RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """
    Train HistGradientBoostingClassifier for binary classification (0=Normal, 1=Attack).
    Uses class_weight='balanced' to account for class representation.
    """
    print(f"      Training binary classifier on {X_train_transformed.shape[0]:,} records ({X_train_transformed.shape[1]} features)...")

    model = HistGradientBoostingClassifier(
        max_iter=config.HGB_MAX_ITER,
        learning_rate=config.HGB_LEARNING_RATE,
        max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
        class_weight="balanced",
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train_transformed, y_train_binary)

    # Save model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.SUSPICIOUS_CLASSIFIER_PATH)
    print(f"      Saved suspicious classifier to: {config.SUSPICIOUS_CLASSIFIER_PATH}")

    return model


def evaluate_binary_classifier(
    model: HistGradientBoostingClassifier,
    X_test_transformed: np.ndarray,
    y_test_binary: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate binary classifier on the real test set, save metrics, confusion matrix,
    and feature importance visualization.
    """
    print("      Evaluating binary classifier on real test set...")
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    acc = float(accuracy_score(y_test_binary, y_pred))
    prec = float(precision_score(y_test_binary, y_pred, zero_division=0))
    rec = float(recall_score(y_test_binary, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_binary, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test_binary, y_proba))

    cls_report_dict = classification_report(
        y_test_binary,
        y_pred,
        target_names=["Normal (0)", "Attack (1)"],
        output_dict=True,
        zero_division=0,
    )
    cls_report_text = classification_report(
        y_test_binary,
        y_pred,
        target_names=["Normal (0)", "Attack (1)"],
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test_binary, y_pred)

    # Save metrics JSON
    metrics = {
        "model_type": "HistGradientBoostingClassifier (Binary)",
        "test_records_evaluated": len(y_test_binary),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "classification_report": {
            "normal": {
                "precision": round(cls_report_dict["Normal (0)"]["precision"], 4),
                "recall": round(cls_report_dict["Normal (0)"]["recall"], 4),
                "f1_score": round(cls_report_dict["Normal (0)"]["f1-score"], 4),
                "support": int(cls_report_dict["Normal (0)"]["support"]),
            },
            "attack": {
                "precision": round(cls_report_dict["Attack (1)"]["precision"], 4),
                "recall": round(cls_report_dict["Attack (1)"]["recall"], 4),
                "f1_score": round(cls_report_dict["Attack (1)"]["f1-score"], 4),
                "support": int(cls_report_dict["Attack (1)"]["support"]),
            },
        },
    }

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BINARY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save text report
    with open(config.BINARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("ThreatLens AI - Binary Threat Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(cls_report_text + "\n\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Binary Threat Confusion Matrix")
    plt.colorbar()
    classes = ["Normal (0)", "Attack (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val:,}",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )

    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig(config.BINARY_CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    # Feature Importance via native permutation importance
    if feature_names:
        print("      Computing feature importances via native scikit-learn permutation...")
        explainer = ModelExplainer(model, feature_names)
        ranked = explainer.compute_global_importance(
            X_val=X_test_transformed,
            y_val=y_test_binary,
            n_repeats=3,
            max_samples=2500,
            random_state=config.RANDOM_STATE,
        )

        top_15 = ranked[:15]
        feat_labels = [item["feature"] for item in top_15][::-1]
        feat_scores = [item["importance_mean"] for item in top_15][::-1]

        plt.figure(figsize=(9, 6))
        plt.barh(feat_labels, feat_scores, color="#1f77b4")
        plt.title("Top 15 Most Discriminative Features (Permutation F1 Importance)")
        plt.xlabel("Importance Mean (Decrease in F1 Score)")
        plt.tight_layout()
        plt.savefig(config.BINARY_FEATURE_IMPORTANCE_PATH, dpi=150)
        plt.close()
        print(f"      Saved feature importance plot: {config.BINARY_FEATURE_IMPORTANCE_PATH}")

    print(f"      Saved binary metrics: {config.BINARY_METRICS_PATH}")
    print(f"      Saved binary report:  {config.BINARY_REPORT_PATH}")
    print(f"      Saved confusion matrix: {config.BINARY_CONFUSION_MATRIX_PATH}")

    return metrics


def predict_suspicious(
    flow: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Predict whether an incoming network flow is suspicious / malicious attack traffic.
    
    Parameters:
        flow: Network flow as a dictionary of raw features, DataFrame row, or preprocessed array.
        
    Returns:
        Dict:
        {
            "is_suspicious": bool,
            "suspicious_probability": float
        }
    """
    model = get_suspicious_classifier()

    if isinstance(flow, (dict, pd.Series, pd.DataFrame)):
        if isinstance(flow, dict):
            df_flow = pd.DataFrame([flow])
        elif isinstance(flow, pd.Series):
            df_flow = pd.DataFrame([flow.to_dict()])
        else:
            df_flow = flow.copy()

        cols_to_drop = [c for c in config.DROP_COLUMNS if c in df_flow.columns]
        if cols_to_drop:
            df_flow = df_flow.drop(columns=cols_to_drop)

        for col in config.CATEGORICAL_FEATURES:
            if col in df_flow.columns:
                df_flow[col] = df_flow[col].astype(str).str.strip()
            else:
                df_flow[col] = "-"
        for col in config.NUMERICAL_FEATURES:
            if col in df_flow.columns:
                df_flow[col] = pd.to_numeric(df_flow[col], errors="coerce")
            else:
                df_flow[col] = 0.0

        preprocessor = get_preprocessor()
        flow_vector = preprocessor.transform(df_flow)
    elif isinstance(flow, np.ndarray):
        flow_vector = flow.reshape(1, -1) if flow.ndim == 1 else flow
    else:
        raise TypeError(f"Unsupported flow data type: {type(flow)}")

    pred = int(model.predict(flow_vector)[0])
    proba = float(model.predict_proba(flow_vector)[0, 1])

    return {
        "is_suspicious": bool(pred == 1),
        "suspicious_probability": round(proba, 4),
    }
