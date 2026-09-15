"""
ThreatLens AI - Deterministic Security Risk Engine
===================================================

Deterministic, non-LLM risk scoring engine that computes unified threat severity
based on binary suspicion probability, unsupervised anomaly deviation, and
multiclass attack taxonomy.

Exact Risk Calculation Formula:
-------------------------------
1. Base Component Score (0 - 90):
   base_score = (
       0.40 * suspicious_probability * 100
       + 0.25 * normalized_anomaly_score * 100
       + 0.25 * attack_probability * 100
   )

2. Attack-Type Severity Adjustment (0 - 10):
   Specific attack vectors represent higher weaponization risk:
   - Worms:          +10 (Self-propagating network traversal)
   - Shellcode:      +9  (Direct code execution / payload delivery)
   - Backdoor:       +8  (Persistent unauthorized command & control)
   - Exploits:       +6  (Vulnerability weaponization)
   - DoS:            +5  (Service disruption / resource exhaustion)
   - Generic:        +4  (Cryptographic / generic collision traffic)
   - Fuzzers:        +3  (State fuzzing / probe testing)
   - Reconnaissance: +2  (Port scanning / host enumeration)
   - Analysis:       +2  (Evasion probe / reconnaissance)
   - Normal:          0  (Benign baseline)

3. Final Clamping:
   final_risk_score = round(clamp(base_score + attack_adjustment, 0, 100))

4. Categorical Severity Tiers:
   - 0  - 24  : LOW
   - 25 - 49  : MEDIUM
   - 50 - 74  : HIGH
   - 75 - 100 : CRITICAL
"""

from typing import Dict, Any, Union

# Attack-type severity adjustments (0 to 10 scale)
ATTACK_SEVERITY_WEIGHTS: Dict[str, float] = {
    "Worms": 10.0,
    "Shellcode": 9.0,
    "Backdoor": 8.0,
    "Exploits": 6.0,
    "DoS": 5.0,
    "Generic": 4.0,
    "Fuzzers": 3.0,
    "Reconnaissance": 2.0,
    "Analysis": 2.0,
    "Normal": 0.0,
}


def calculate_risk(
    suspicious_probability: float,
    anomaly_score: float,
    attack_probability: float,
    attack_type: str = "Normal",
) -> Dict[str, Union[int, str]]:
    """
    Calculate deterministic threat risk score and categorical severity tier.

    Parameters:
        suspicious_probability (float): Binary threat likelihood (0.0 - 1.0).
        anomaly_score (float): Normalized unsupervised outlier score (0.0 - 1.0).
        attack_probability (float): Multiclass attack classification confidence (0.0 - 1.0).
        attack_type (str): Identified attack category name or 'Normal'.

    Returns:
        Dict:
        {
            "risk_score": int (0 - 100),
            "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        }
    """
    # Defensive normalization and bounds clamping
    p_susp = max(0.0, min(1.0, float(suspicious_probability)))
    p_anom = max(0.0, min(1.0, float(anomaly_score)))
    p_attk = max(0.0, min(1.0, float(attack_probability)))

    # Step 1: Base Component Calculation
    base_score = (
        0.40 * p_susp * 100.0
        + 0.25 * p_anom * 100.0
        + 0.25 * p_attk * 100.0
    )

    # Step 2: Attack-Type Severity Adjustment
    # Adjustment only applies if the flow has positive attack probability or suspicion
    attack_name = str(attack_type).strip()
    if p_susp > 0.5 or p_attk > 0.0:
        adjustment = ATTACK_SEVERITY_WEIGHTS.get(attack_name, 3.0)
    else:
        adjustment = 0.0

    raw_score = base_score + adjustment

    # Step 3: Clamping to 0 - 100 range
    risk_score = int(round(max(0.0, min(100.0, raw_score))))

    # Step 4: Determine Severity Tier
    if risk_score < 25:
        severity = "LOW"
    elif risk_score < 50:
        severity = "MEDIUM"
    elif risk_score < 75:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    return {
        "risk_score": risk_score,
        "severity": severity,
    }
