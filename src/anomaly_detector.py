"""
ThreatLens AI - Anomaly Detection Engine (Isolation Forest)
===========================================================

Role:
Identifies zero-day anomalies, novel polymorphic variations, and unusual
network traffic patterns by training exclusively on benign (normal) traffic profiles.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, Optional
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config

# Global cached instances for fast inference
_PREPROCESSOR = None
_ANOMALY_DETECTOR = None


def get_preprocessor():
    """Load or return cached preprocessor pipeline."""
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {config.PREPROCESSOR_PATH}")
        _PREPROCESSOR = joblib.load(config.PREPROCESSOR_PATH)
    return _PREPROCESSOR


def get_anomaly_detector():
    """Load or return cached Isolation Forest model."""
    global _ANOMALY_DETECTOR
    if _ANOMALY_DETECTOR is None:
        if not config.ANOMALY_DETECTOR_PATH.exists():
            raise FileNotFoundError(f"Anomaly detector not found at {config.ANOMALY_DETECTOR_PATH}")
        _ANOMALY_DETECTOR = joblib.load(config.ANOMALY_DETECTOR_PATH)
    return _ANOMALY_DETECTOR


def train_anomaly_detector(
    X_train_transformed: np.ndarray,
    y_train_binary: np.ndarray,
    contamination: float = config.ISOLATION_FOREST_CONTAMINATION,
    random_state: int = config.RANDOM_STATE,
) -> IsolationForest:
    """
    Train Isolation Forest exclusively on BENIGN traffic samples.
    
    Parameters:
        X_train_transformed (np.ndarray): Preprocessed training feature matrix.
        y_train_binary (np.ndarray): Binary labels (0 = Normal, 1 = Attack).
        contamination (float): Expected outlier proportion in baseline benign traffic.
        random_state (int): Reproducibility seed.
        
    Returns:
        model (IsolationForest): Trained model.
    """
    # Filter strictly for Benign / Normal samples (label == 0)
    benign_mask = (y_train_binary == 0)
    X_benign = X_train_transformed[benign_mask]

    print(f"      Training IsolationForest on {X_benign.shape[0]:,} benign baseline flows...")

    model = IsolationForest(
        n_estimators=config.ISOLATION_FOREST_N_ESTIMATORS,
        max_samples=config.ISOLATION_FOREST_MAX_SAMPLES,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_benign)

    # Save model
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.ANOMALY_DETECTOR_PATH)
    print(f"      Saved anomaly detector to: {config.ANOMALY_DETECTOR_PATH}")

    return model


def evaluate_anomaly_detector(
    model: IsolationForest,
    X_test_transformed: np.ndarray,
    y_test_binary: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate Isolation Forest performance against the real test ground-truth.
    Note: Distinguishes unsupervised anomaly detection metrics from supervised classification.
    """
    # IsolationForest.predict: +1 for inliers (normal), -1 for outliers (anomalies)
    preds = model.predict(X_test_transformed)
    pred_anomalies = (preds == -1).astype(int)

    # decision_function: lower values indicate higher anomaly likelihood
    raw_scores = model.decision_function(X_test_transformed)
    
    # Invert and normalize so that higher scores represent higher anomaly probability
    # Typical decision_function range is approx [-0.5, 0.5]
    anomaly_scores = 0.5 - raw_scores
    anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

    # Evaluation metrics
    tp = int(((pred_anomalies == 1) & (y_test_binary == 1)).sum())
    fp = int(((pred_anomalies == 1) & (y_test_binary == 0)).sum())
    tn = int(((pred_anomalies == 0) & (y_test_binary == 0)).sum())
    fn = int(((pred_anomalies == 0) & (y_test_binary == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "model_type": "IsolationForest",
        "contamination": model.contamination,
        "test_records_evaluated": len(y_test_binary),
        "anomalies_flagged": int(pred_anomalies.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "anomaly_precision": round(precision, 4),
        "anomaly_recall_attack_detection_rate": round(recall, 4),
        "anomaly_f1": round(f1, 4),
        "note": "Unsupervised baseline; trained solely on normal traffic patterns."
    }
    return metrics


def detect_anomaly(flow: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Predict whether an incoming network flow is an anomaly.
    
    Parameters:
        flow: Network flow as a dictionary of raw features, a DataFrame row, or preprocessed array.
        
    Returns:
        Dict:
        {
            "is_anomaly": bool,
            "anomaly_score": float
        }
    """
    model = get_anomaly_detector()

    # Preprocess if raw feature dictionary or DataFrame
    if isinstance(flow, (dict, pd.Series, pd.DataFrame)):
        if isinstance(flow, dict):
            df_flow = pd.DataFrame([flow])
        elif isinstance(flow, pd.Series):
            df_flow = pd.DataFrame([flow.to_dict()])
        else:
            df_flow = flow.copy()

        # Drop identifier or target columns if present
        cols_to_drop = [c for c in config.DROP_COLUMNS if c in df_flow.columns]
        if cols_to_drop:
            df_flow = df_flow.drop(columns=cols_to_drop)

        # Ensure all expected raw features exist
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

    pred = model.predict(flow_vector)[0]
    decision = model.decision_function(flow_vector)[0]

    # Convert decision score to a normalized 0.0 - 1.0 anomaly score
    # decision_function: negative is outlier, positive is inlier
    normalized_score = float(np.clip(0.5 - decision, 0.0, 1.0))
    is_anomaly = bool(pred == -1)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(normalized_score, 4),
    }
