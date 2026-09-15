"""
ThreatLens AI - Unified Threat Predictor & Security Event Interface
===================================================================

The primary public-facing inference interface for ThreatLens AI.
Designed for seamless, direct integration with FastAPI backends, LangGraph agents,
and real-time SOC monitoring dashboards.

Public Import:
--------------
from src.predictor import predict_threat

result = predict_threat(flow)

Output Schema (Standardized SecurityEvent):
-------------------------------------------
{
    "event_id": "EVT-XXXX",
    "timestamp": "ISO timestamp",

    "source_ip": "...",
    "destination_ip": "...",
    "protocol": "...",

    "is_suspicious": bool,
    "suspicious_probability": float,

    "is_anomaly": bool,
    "anomaly_score": float,

    "attack_type": str,
    "attack_probability": float,

    "risk_score": int (0-100),
    "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",

    "explanation": {
        "method": str,
        "top_features": [
            {
                "feature": str,
                "importance": float,
                "direction": str
            }
        ]
    }
}
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Union, Optional

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.risk_engine import calculate_risk
from src.explainability import explain_prediction

# Global cache for serialized models & artifacts
_MODELS_CACHE: Dict[str, Any] = {}


def _load_model_artifacts() -> Dict[str, Any]:
    """Load and cache all required model artifacts for low-latency inference."""
    global _MODELS_CACHE

    if "preprocessor" not in _MODELS_CACHE:
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {config.PREPROCESSOR_PATH}")
        _MODELS_CACHE["preprocessor"] = joblib.load(config.PREPROCESSOR_PATH)

    if "binary_classifier" not in _MODELS_CACHE:
        if not config.SUSPICIOUS_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Binary classifier not found at {config.SUSPICIOUS_CLASSIFIER_PATH}")
        _MODELS_CACHE["binary_classifier"] = joblib.load(config.SUSPICIOUS_CLASSIFIER_PATH)

    if "attack_classifier" not in _MODELS_CACHE:
        if not config.ATTACK_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Attack classifier not found at {config.ATTACK_CLASSIFIER_PATH}")
        _MODELS_CACHE["attack_classifier"] = joblib.load(config.ATTACK_CLASSIFIER_PATH)

    if "anomaly_detector" not in _MODELS_CACHE:
        if not config.ANOMALY_DETECTOR_PATH.exists():
            raise FileNotFoundError(f"Anomaly detector not found at {config.ANOMALY_DETECTOR_PATH}")
        _MODELS_CACHE["anomaly_detector"] = joblib.load(config.ANOMALY_DETECTOR_PATH)

    if "attack_classes" not in _MODELS_CACHE:
        if not config.ATTACK_CLASSES_PATH.exists():
            raise FileNotFoundError(f"Attack classes metadata not found at {config.ATTACK_CLASSES_PATH}")
        import json
        with open(config.ATTACK_CLASSES_PATH, "r", encoding="utf-8") as f:
            _MODELS_CACHE["attack_classes"] = json.load(f)["classes"]

    return _MODELS_CACHE


def predict_threat(flow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified threat detection pipeline analyzing a single network telemetry flow.

    Parameters:
        flow (Dict[str, Any]): Raw network telemetry containing flow attributes
                               (UNSW-NB15 format: dur, proto, service, state, spkts, etc.).

    Returns:
        Dict[str, Any]: Standardized SecurityEvent object containing:
          - Telemetry metadata (event_id, timestamp, IPs, protocol)
          - Binary threat assessment (is_suspicious, suspicious_probability)
          - Anomaly assessment (is_anomaly, anomaly_score)
          - Multiclass taxonomy (attack_type, attack_probability)
          - Unified risk score & categorical severity
          - Feature-level explainability
    """
    # 1. Input Validation
    if not isinstance(flow, dict):
        raise TypeError(f"Expected flow to be a dictionary, got {type(flow).__name__}")
    if not flow:
        raise ValueError("Network flow dictionary cannot be empty")

    # Load cached pipeline artifacts
    models = _load_model_artifacts()
    preprocessor = models["preprocessor"]
    binary_clf = models["binary_classifier"]
    attack_clf = models["attack_classifier"]
    anomaly_det = models["anomaly_detector"]
    attack_classes = models["attack_classes"]

    # 2. Extract Network Metadata
    event_id = str(flow.get("event_id") or f"EVT-{uuid.uuid4().hex[:8].upper()}")
    timestamp = str(flow.get("timestamp") or datetime.now(timezone.utc).isoformat())
    source_ip = str(flow.get("source_ip") or flow.get("srcip") or flow.get("src_ip") or "192.168.1.100")
    destination_ip = str(flow.get("destination_ip") or flow.get("dstip") or flow.get("dst_ip") or "10.0.0.50")
    protocol = str(flow.get("protocol") or flow.get("proto") or "tcp").strip()

    # 3. Convert dictionary into sanitized single-row DataFrame
    df_flow = pd.DataFrame([flow])

    # Strip identifiers or target columns if present to prevent leakage
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df_flow.columns]
    if cols_to_drop:
        df_flow = df_flow.drop(columns=cols_to_drop)

    # Ensure all expected categorical and numerical columns exist
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

    # 4. Apply Fitted Preprocessor Pipeline
    flow_vector = preprocessor.transform(df_flow)
    if hasattr(flow_vector, "toarray"):
        flow_vector = flow_vector.toarray()

    # 5. Binary Threat Classifier
    binary_proba = float(binary_clf.predict_proba(flow_vector)[0, 1])
    is_suspicious = bool(binary_proba >= 0.5)

    # 6. Isolation Forest Anomaly Detection
    raw_decision = float(anomaly_det.decision_function(flow_vector)[0])
    raw_pred = int(anomaly_det.predict(flow_vector)[0])
    is_anomaly = bool(raw_pred == -1)
    # Normalized score: inliers are positive decision, outliers negative
    # Scale to 0.0 - 1.0 where 1.0 represents extreme deviation
    normalized_anomaly_score = float(np.clip(0.5 - raw_decision, 0.0, 1.0))

    # 7. Attack-Type Classification
    if is_suspicious:
        # Evaluate multiclass attack family
        attack_probs = attack_clf.predict_proba(flow_vector)[0]
        pred_attack_idx = int(np.argmax(attack_probs))
        attack_type = str(attack_classes[pred_attack_idx])
        attack_probability = float(attack_probs[pred_attack_idx])
    else:
        # Benign traffic is cleanly designated as Normal with 0 attack probability
        attack_type = "Normal"
        attack_probability = 0.0

    # 8. Calculate Deterministic Risk Score
    risk_assessment = calculate_risk(
        suspicious_probability=binary_proba,
        anomaly_score=normalized_anomaly_score,
        attack_probability=attack_probability,
        attack_type=attack_type,
    )

    # 9. Generate Explainability
    explanation = explain_prediction(flow=flow, top_n=5)

    # 10. Assemble Standardized SecurityEvent
    security_event = {
        "event_id": event_id,
        "timestamp": timestamp,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "protocol": protocol,
        "is_suspicious": is_suspicious,
        "suspicious_probability": round(binary_proba, 4),
        "is_anomaly": is_anomaly,
        "anomaly_score": round(normalized_anomaly_score, 4),
        "attack_type": attack_type,
        "attack_probability": round(attack_probability, 4),
        "risk_score": int(risk_assessment["risk_score"]),
        "severity": str(risk_assessment["severity"]),
        "explanation": explanation,
    }

    return security_event


if __name__ == "__main__":
    # Test execution using sample flow
    test_flow = {
        "dur": 0.000009,
        "proto": "udp",
        "service": "dns",
        "state": "INT",
        "spkts": 2,
        "dpkts": 0,
        "sbytes": 114,
        "dbytes": 0,
        "rate": 111111.1072,
        "sttl": 254,
        "dttl": 0,
        "sload": 50666664.0,
        "dload": 0.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.009,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 57,
        "dmean": 0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 2,
        "ct_state_ttl": 2,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 2,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 1,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 0,
    }
    result = predict_threat(test_flow)
    import pprint
    pprint.pprint(result)
