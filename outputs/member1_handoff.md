# ThreatLens AI — Member 1 ML Handoff

This document details the Machine Learning and Threat Detection architecture, integration interface, and real benchmark metrics for the ThreatLens AI pipeline.

---

## Dataset

**UNSW-NB15 Network Intrusion Dataset**

- **Training records**: `173,435` valid records after removing the single truncated record (`id = 173436`).
- **Testing records**: `82,332`
- **Original ML features**: `42` (39 numerical attributes, 3 categorical attributes)
- **Transformed features**: `194` (via fitted scikit-learn `ColumnTransformer`)

---

## Models

### 1. Binary Classifier (`HistGradientBoostingClassifier`)
- **Artifact**: `models/suspicious_classifier.pkl`
- **Purpose**: First-line triage determining **Normal (0)** vs **Attack (1)** traffic.
- **Handling**: Configured with `class_weight='balanced'` to prevent majority class bias without distorting raw feature space.

### 2. Attack Classifier (`HistGradientBoostingClassifier`)
- **Artifact**: `models/attack_classifier.pkl`
- **Classes**: `models/attack_classes.json`
- **Purpose**: 9-class attack family taxonomy classification (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms).
- **Rule**: "Normal" is strictly excluded from attack family classification.

### 3. Anomaly Detector (`IsolationForest`)
- **Artifact**: `models/anomaly_detector.pkl`
- **Purpose**: Zero-day behavioral anomaly detection trained exclusively on benign (`Normal`) baseline network traffic.

---

## REAL TEST METRICS

All metrics below were evaluated strictly against the untouched, official test partition (`82,332 real test flows`). Zero metrics are synthetic or fabricated.

### Binary Classification Metrics:
- **Accuracy**: `90.62%`
- **Precision**: `87.67%`
- **Recall**: `96.54%` (caught 43,763 out of 45,332 test attack flows)
- **F1-Score**: `91.89%` (`0.9189`)
- **ROC-AUC**: `98.46%` (`0.9846`)

### Multiclass Attack-Family Metrics (Evaluated on 45,332 Attack Test Flows):
- **Accuracy**: `74.22%`
- **Macro Precision**: `54.27%`
- **Macro Recall**: `63.49%`
- **Macro F1**: `53.59%` (`0.5359`)
- **Weighted F1**: `78.24%` (`0.7824`)

### Anomaly Detection Metrics (Unsupervised Baseline):
- **Precision**: `85.13%`
- **Attack Recall (Catch Rate)**: `28.06%`

---

## Integration

The teammate (FastAPI / Agent / Dashboard developer) should use **ONLY** the unified predictor:

```python
from src.predictor import predict_threat

result = predict_threat(flow)
```

> **IMPORTANT**: Do **not** duplicate ML logic, preprocessing, or feature scaling inside FastAPI or the agent. The `predict_threat()` function encapsulates the entire inference pipeline internally.

---

## SecurityEvent Output Schema

The function returns a standardized dictionary with the following 14 fields:

| Field | Type | Description |
| :--- | :---: | :--- |
| `event_id` | `str` | Unique security event identifier (e.g. `"EVT-962F48AF"`). |
| `timestamp` | `str` | ISO 8601 UTC timestamp of event processing. |
| `source_ip` | `str` | Source IP address extracted from telemetry or flow header. |
| `destination_ip` | `str` | Destination IP address extracted from telemetry or flow header. |
| `protocol` | `str` | Transport / application protocol (e.g. `"tcp"`, `"udp"`, `"ospf"`). |
| `is_suspicious` | `bool` | `True` if binary probability $\ge 0.50$, else `False`. |
| `suspicious_probability` | `float` | Model confidence of malicious activity (`0.0000` to `1.0000`). |
| `is_anomaly` | `bool` | `True` if Isolation Forest flags the flow as an outlier, else `False`. |
| `anomaly_score` | `float` | Normalized behavioral anomaly score (`0.0000` to `1.0000`). |
| `attack_type` | `str` | Identified attack category name (or `"Normal"` if benign). |
| `attack_probability` | `float` | Multiclass model confidence (`0.0` if `"Normal"`). |
| `risk_score` | `int` | Deterministic threat score from `0` to `100`. |
| `severity` | `str` | Categorical risk tier: `"LOW"`, `"MEDIUM"`, `"HIGH"`, or `"CRITICAL"`. |
| `explanation` | `dict` | Method used and top 5 contributing features with directions. |

---

## Risk Scoring Formula

Risk is **completely deterministic** and computed by [`src/risk_engine.py`](file:///c:/Users/kbhan/OneDrive/Desktop/ThreatLens-AI/src/risk_engine.py):

$$\text{base\_score} = 0.40 \times (\text{suspicious\_probability} \times 100) + 0.25 \times (\text{anomaly\_score} \times 100) + 0.25 \times (\text{attack\_probability} \times 100)$$

Followed by an attack-type severity adjustment:
- **Worms**: $+10$
- **Shellcode**: $+9$
- **Backdoor**: $+8$
- **Exploits**: $+6$
- **DoS**: $+5$
- **Generic**: $+4$
- **Fuzzers**: $+3$
- **Reconnaissance**: $+2$
- **Analysis**: $+2$
- **Normal**: $0$

Clamped between `0` and `100`:

- `0` – `24`   : **LOW**
- `25` – `49`  : **MEDIUM**
- `50` – `74`  : **HIGH**
- `75` – `100` : **CRITICAL**

> **CRITICAL RULE**: The LLM / Agent must **NOT** calculate or override the risk score. Risk calculation is strictly mathematical and deterministic.

---

## Important Limitations

1. **Recall-Optimized Binary Classifier**:
   The binary threat classifier intentionally prioritizes attack recall (`96.54%`) to prevent critical breaches. Consequently, it exhibits a ~16.6% false positive rate on complex normal flows.
2. **Behavioral Anomaly Detector**:
   The Isolation Forest was trained solely on benign flows to detect zero-days without labeled supervision. It is intended as an auxiliary behavioral signal and should **not** be used as the sole standalone detector.
3. **Imbalanced Rare Attack Families**:
   The multiclass classifier achieves high accuracy on dominant attack families (Generic F1: 98.46%, Reconnaissance F1: 85.62%, Fuzzers F1: 76.75%), but rare classes (Worms, Backdoor) have higher variance due to severe training imbalance.
4. **Hackathon Scope**:
   Do **not** claim full production readiness. These models are calibrated for demonstration on UNSW-NB15 benchmark telemetry and require live domain adaptation prior to real enterprise perimeter deployment.
