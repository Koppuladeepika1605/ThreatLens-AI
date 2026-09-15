"""
ThreatLens AI - Multiclass Attack Classifier (HistGradientBoostingClassifier)
=============================================================================

Role:
Specialized classification engine that taxonomizes verified threat flows into
specific attack families:
  1. Generic
  2. Exploits
  3. Fuzzers
  4. DoS
  5. Reconnaissance
  6. Analysis
  7. Backdoor
  8. Shellcode
  9. Worms

Crucial Rules:
- "Normal" is NOT an attack class and is strictly excluded from training and evaluation.
- Class imbalance is addressed via balanced sample weighting.
- Evaluated strictly against attack samples in the real test partition.
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
    classification_report,
    confusion_matrix,
)

from src import config

# Global cached model and metadata
_ATTACK_CLASSIFIER = None
_ATTACK_CLASSES = None
_PREPROCESSOR = None


def get_preprocessor():
    """Load or return cached preprocessor pipeline."""
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {config.PREPROCESSOR_PATH}")
        _PREPROCESSOR = joblib.load(config.PREPROCESSOR_PATH)
    return _PREPROCESSOR


def get_attack_classifier():
    """Load or return cached attack classifier model."""
    global _ATTACK_CLASSIFIER
    if _ATTACK_CLASSIFIER is None:
        if not config.ATTACK_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Attack classifier not found at {config.ATTACK_CLASSIFIER_PATH}")
        _ATTACK_CLASSIFIER = joblib.load(config.ATTACK_CLASSIFIER_PATH)
    return _ATTACK_CLASSIFIER


def get_attack_classes() -> List[str]:
    """Load or return cached attack class names."""
    global _ATTACK_CLASSES
    if _ATTACK_CLASSES is None:
        if not config.ATTACK_CLASSES_PATH.exists():
            raise FileNotFoundError(f"Attack classes metadata not found at {config.ATTACK_CLASSES_PATH}")
        with open(config.ATTACK_CLASSES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            _ATTACK_CLASSES = data["classes"]
    return _ATTACK_CLASSES


def train_attack_classifier(
    X_train_transformed: np.ndarray,
    y_train_attack_series: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[HistGradientBoostingClassifier, List[str], Dict[str, int]]:
    """
    Train HistGradientBoostingClassifier exclusively on ATTACK samples.
    
    Parameters:
        X_train_transformed (np.ndarray): Transformed training features.
        y_train_attack_series (pd.Series): Raw attack_cat labels.
        random_state (int): Reproducibility seed.
        
    Returns:
        model, attack_classes, class_to_idx
    """
    # 1. Filter out 'Normal'
    attack_mask = (y_train_attack_series != "Normal").to_numpy()
    X_attacks = X_train_transformed[attack_mask]
    y_attacks_raw = y_train_attack_series[attack_mask].values

    # Determine unique attack categories
    attack_classes = sorted(list(set(y_attacks_raw)))
    class_to_idx = {cls: idx for idx, cls in enumerate(attack_classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(attack_classes)}
    y_attacks_encoded = np.array([class_to_idx[c] for c in y_attacks_raw])

    print(f"      Training multiclass attack classifier on {len(y_attacks_encoded):,} attack samples...")
    print(f"      Attack classes ({len(attack_classes)}): {attack_classes}")

    # Build and fit HistGradientBoostingClassifier with balanced class weighting
    model = HistGradientBoostingClassifier(
        max_iter=config.HGB_MAX_ITER,
        learning_rate=config.HGB_LEARNING_RATE,
        max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
        class_weight="balanced",
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_attacks, y_attacks_encoded)

    # Save model and metadata
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.ATTACK_CLASSIFIER_PATH)

    metadata = {
        "num_classes": len(attack_classes),
        "classes": attack_classes,
        "class_to_index": class_to_idx,
        "index_to_class": idx_to_class,
    }
    with open(config.ATTACK_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"      Saved attack classifier to: {config.ATTACK_CLASSIFIER_PATH}")
    print(f"      Saved attack class metadata to: {config.ATTACK_CLASSES_PATH}")

    return model, attack_classes, class_to_idx


def evaluate_attack_classifier(
    model: HistGradientBoostingClassifier,
    attack_classes: List[str],
    class_to_idx: Dict[str, int],
    X_test_transformed: np.ndarray,
    y_test_attack_series: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate attack-type classifier strictly against attack samples in the real test set.
    """
    # Filter test set for attack samples only (exclude Normal)
    test_attack_mask = (y_test_attack_series != "Normal").to_numpy()
    X_test_attacks = X_test_transformed[test_attack_mask]
    y_test_attacks_raw = y_test_attack_series[test_attack_mask].values
    y_test_attacks_encoded = np.array([class_to_idx[c] for c in y_test_attacks_raw])

    y_pred = model.predict(X_test_attacks)

    acc = float(accuracy_score(y_test_attacks_encoded, y_pred))
    macro_prec = float(precision_score(y_test_attacks_encoded, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test_attacks_encoded, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_attacks_encoded, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_attacks_encoded, y_pred, average="weighted", zero_division=0))

    cls_report_dict = classification_report(
        y_test_attacks_encoded,
        y_pred,
        target_names=attack_classes,
        output_dict=True,
        zero_division=0,
    )
    cls_report_text = classification_report(
        y_test_attacks_encoded,
        y_pred,
        target_names=attack_classes,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test_attacks_encoded, y_pred)

    # Save metrics JSON
    metrics = {
        "model_type": "HistGradientBoostingClassifier (Multiclass)",
        "test_attack_samples_evaluated": len(y_test_attacks_encoded),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_metrics": {
            cls_name: {
                "precision": round(cls_report_dict[cls_name]["precision"], 4),
                "recall": round(cls_report_dict[cls_name]["recall"], 4),
                "f1_score": round(cls_report_dict[cls_name]["f1-score"], 4),
                "support": int(cls_report_dict[cls_name]["support"]),
            }
            for cls_name in attack_classes
        },
    }
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ATTACK_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report text
    with open(config.ATTACK_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("ThreatLens AI - Attack Category Classification Report\n")
        f.write("=" * 65 + "\n")
        f.write(cls_report_text)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Attack Category Confusion Matrix (Test Set)")
    plt.colorbar()
    tick_marks = np.arange(len(attack_classes))
    plt.xticks(tick_marks, attack_classes, rotation=45, ha="right")
    plt.yticks(tick_marks, attack_classes)

    # Format numbers inside matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val:,}",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.ylabel("True Attack Category")
    plt.xlabel("Predicted Attack Category")
    plt.savefig(config.ATTACK_CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    print(f"      Saved attack metrics to: {config.ATTACK_METRICS_PATH}")
    print(f"      Saved attack report to:  {config.ATTACK_REPORT_PATH}")
    print(f"      Saved confusion matrix:  {config.ATTACK_CONFUSION_MATRIX_PATH}")

    return metrics


def predict_attack_type(
    flow: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Classify an attack flow into its specific attack taxonomy category.
    
    Parameters:
        flow: Network flow as a dictionary of raw features, DataFrame row, or preprocessed array.
        
    Returns:
        Dict:
        {
            "attack_type": str,
            "attack_probability": float
        }
    """
    model = get_attack_classifier()
    attack_classes = get_attack_classes()

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

    pred_idx = int(model.predict(flow_vector)[0])
    probs = model.predict_proba(flow_vector)[0]
    predicted_class = attack_classes[pred_idx]
    class_prob = float(probs[pred_idx])

    return {
        "attack_type": predicted_class,
        "attack_probability": round(class_prob, 4),
    }
