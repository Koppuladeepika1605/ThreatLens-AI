"""
ThreatLens AI - Master Machine Learning Training Pipeline
==========================================================

Orchestrates the complete offline ML training and evaluation workflow:
  1. Load cleaned real UNSW-NB15 dataset (drops corrupt row id=173436).
  2. Load pre-fitted ColumnTransformer (never refits on test data).
  3. Transform X_train and X_test into 194-dimensional feature space.
  4. Train & evaluate Binary Threat Classifier (HistGradientBoostingClassifier).
  5. Train & evaluate Multiclass Attack Classifier (HistGradientBoostingClassifier).
  6. Train & evaluate Anomaly Detection Engine (IsolationForest).
  7. Validate end-to-end inference flow using a real test telemetry sample.
  8. Generate comprehensive summary report in outputs/ml_training_summary.txt.
"""

import sys
import gc
import json
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.preprocessing import load_and_clean_data, prepare_features
from src.train_classifier import (
    train_binary_classifier,
    evaluate_binary_classifier,
    predict_suspicious,
)
from src.attack_classifier import (
    train_attack_classifier,
    evaluate_attack_classifier,
    predict_attack_type,
)
from src.anomaly_detector import (
    train_anomaly_detector,
    evaluate_anomaly_detector,
    detect_anomaly,
)


def run_full_training_pipeline() -> Dict[str, Any]:
    """Execute the full end-to-end training and validation pipeline."""
    start_time = time.time()
    print("=" * 85)
    print("        THREATLENS AI - END-TO-END MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 85)
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Models Dir:   {config.MODELS_DIR}")
    print(f"Outputs Dir:  {config.OUTPUTS_DIR}")
    print()

    # -------------------------------------------------------------------------
    # STEP 1: LOAD CLEANED REAL DATA
    # -------------------------------------------------------------------------
    print("[1/6] Loading cleaned training and testing datasets from CSV...")
    df_train, df_test, orig_train_rows = load_and_clean_data()
    cleaned_train_rows = len(df_train)
    test_rows = len(df_test)
    print(f"      Original train rows: {orig_train_rows:,}")
    print(f"      Cleaned train rows:  {cleaned_train_rows:,} (dropped row id=173436)")
    print(f"      Testing rows:        {test_rows:,}")
    print()

    # -------------------------------------------------------------------------
    # STEP 2: PREPARE FEATURES & TARGETS
    # -------------------------------------------------------------------------
    print("[2/6] Preparing feature matrices and target vectors...")
    (
        X_train_raw,
        y_train_binary,
        y_train_attack,
        X_test_raw,
        y_test_binary,
        y_test_attack,
    ) = prepare_features(df_train, df_test)

    # Free raw DataFrames to preserve memory
    del df_train, df_test
    gc.collect()

    # Load existing fitted preprocessor and feature names
    print("      Loading pre-fitted preprocessor from models/preprocessor.pkl...")
    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor at {config.PREPROCESSOR_PATH}. Run src/preprocessing.py first!")
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)

    with open(config.FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"      Feature catalog contains {len(feature_names)} features.")
    print("      Transforming X_train and X_test...")
    X_train_tf = preprocessor.transform(X_train_raw)
    X_test_tf = preprocessor.transform(X_test_raw)

    # Ensure dense numpy array format
    if hasattr(X_train_tf, "toarray"):
        X_train_tf = X_train_tf.toarray()
    if hasattr(X_test_tf, "toarray"):
        X_test_tf = X_test_tf.toarray()

    print(f"      X_train matrix shape: {X_train_tf.shape}")
    print(f"      X_test matrix shape:  {X_test_tf.shape}")
    print()

    # -------------------------------------------------------------------------
    # STEP 3: TRAIN & EVALUATE BINARY THREAT CLASSIFIER
    # -------------------------------------------------------------------------
    print("[3/6] Component 1: Binary Threat Classifier (HistGradientBoostingClassifier)...")
    binary_model = train_binary_classifier(
        X_train_transformed=X_train_tf,
        y_train_binary=y_train_binary.to_numpy(),
        random_state=config.RANDOM_STATE,
    )
    binary_metrics = evaluate_binary_classifier(
        model=binary_model,
        X_test_transformed=X_test_tf,
        y_test_binary=y_test_binary.to_numpy(),
        feature_names=feature_names,
    )
    print(f"      Binary Accuracy:  {binary_metrics['accuracy'] * 100:.2f}%")
    print(f"      Binary Precision: {binary_metrics['precision'] * 100:.2f}%")
    print(f"      Binary Recall:    {binary_metrics['recall'] * 100:.2f}%")
    print(f"      Binary F1-Score:  {binary_metrics['f1_score']:.4f}")
    print(f"      Binary ROC-AUC:   {binary_metrics['roc_auc']:.4f}")
    print()

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN & EVALUATE MULTICLASS ATTACK CLASSIFIER
    # -------------------------------------------------------------------------
    print("[4/6] Component 2: Multiclass Attack Classifier (HistGradientBoostingClassifier)...")
    attack_model, attack_classes, class_to_idx = train_attack_classifier(
        X_train_transformed=X_train_tf,
        y_train_attack_series=y_train_attack,
        random_state=config.RANDOM_STATE,
    )
    attack_metrics = evaluate_attack_classifier(
        model=attack_model,
        attack_classes=attack_classes,
        class_to_idx=class_to_idx,
        X_test_transformed=X_test_tf,
        y_test_attack_series=y_test_attack,
    )
    print(f"      Attack Accuracy:     {attack_metrics['accuracy'] * 100:.2f}%")
    print(f"      Attack Macro F1:     {attack_metrics['macro_f1']:.4f}")
    print(f"      Attack Weighted F1:  {attack_metrics['weighted_f1']:.4f}")
    print()

    # -------------------------------------------------------------------------
    # STEP 5: TRAIN & EVALUATE ANOMALY DETECTOR
    # -------------------------------------------------------------------------
    print("[5/6] Component 3: Anomaly Detector (IsolationForest on Benign Traffic)...")
    anomaly_model = train_anomaly_detector(
        X_train_transformed=X_train_tf,
        y_train_binary=y_train_binary.to_numpy(),
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_STATE,
    )
    anomaly_metrics = evaluate_anomaly_detector(
        model=anomaly_model,
        X_test_transformed=X_test_tf,
        y_test_binary=y_test_binary.to_numpy(),
    )
    print(f"      Anomaly Recall (Attack Catch Rate): {anomaly_metrics['anomaly_recall_attack_detection_rate'] * 100:.2f}%")
    print(f"      Anomaly Precision:                  {anomaly_metrics['anomaly_precision'] * 100:.2f}%")
    print()

    # -------------------------------------------------------------------------
    # STEP 6: MODEL RE-LOAD & TEST FLOW INFERENCE VALIDATION
    # -------------------------------------------------------------------------
    print("[6/6] Validation: Re-loading all saved models and testing live inference pipeline...")
    # Verify reloading
    loaded_preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    loaded_binary = joblib.load(config.SUSPICIOUS_CLASSIFIER_PATH)
    loaded_attack = joblib.load(config.ATTACK_CLASSIFIER_PATH)
    loaded_anomaly = joblib.load(config.ANOMALY_DETECTOR_PATH)
    print("      [PASS] All 4 serialized models/artifacts reloaded successfully from disk.")

    # Select a real attack flow and a real benign flow from X_test_raw
    test_attack_idx = int(np.where(y_test_binary.to_numpy() == 1)[0][0])
    test_normal_idx = int(np.where(y_test_binary.to_numpy() == 0)[0][0])

    sample_attack_flow = X_test_raw.iloc[test_attack_idx].to_dict()
    true_attack_cat = y_test_attack.iloc[test_attack_idx]

    sample_normal_flow = X_test_raw.iloc[test_normal_idx].to_dict()
    true_normal_cat = y_test_attack.iloc[test_normal_idx]

    print()
    print("      --- LIVE TEST SAMPLE 1 (Known Attack Telemetry) ---")
    print(f"      Ground Truth: label=1 (Attack), attack_cat={true_attack_cat}")
    pred_susp_1 = predict_suspicious(sample_attack_flow)
    pred_anom_1 = detect_anomaly(sample_attack_flow)
    pred_attk_1 = predict_attack_type(sample_attack_flow)
    print(f"      Binary Prediction:  is_suspicious={pred_susp_1['is_suspicious']} (prob={pred_susp_1['suspicious_probability']})")
    print(f"      Anomaly Prediction: is_anomaly={pred_anom_1['is_anomaly']} (score={pred_anom_1['anomaly_score']})")
    print(f"      Attack Prediction:  attack_type={pred_attk_1['attack_type']} (prob={pred_attk_1['attack_probability']})")

    print()
    print("      --- LIVE TEST SAMPLE 2 (Known Normal Telemetry) ---")
    print(f"      Ground Truth: label=0 (Normal), attack_cat={true_normal_cat}")
    pred_susp_2 = predict_suspicious(sample_normal_flow)
    pred_anom_2 = detect_anomaly(sample_normal_flow)
    print(f"      Binary Prediction:  is_suspicious={pred_susp_2['is_suspicious']} (prob={pred_susp_2['suspicious_probability']})")
    print(f"      Anomaly Prediction: is_anomaly={pred_anom_2['is_anomaly']} (score={pred_anom_2['anomaly_score']})")
    print()

    # -------------------------------------------------------------------------
    # GENERATE TRAINING SUMMARY REPORT
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    summary_lines = [
        "=" * 85,
        "           THREATLENS AI - MACHINE LEARNING PIPELINE TRAINING SUMMARY",
        "=" * 85,
        f"Execution Time:                   {elapsed_time:.2f} seconds",
        f"Dataset:                          UNSW-NB15 Network Intrusion Dataset",
        f"Training rows after cleaning:     {cleaned_train_rows:,}",
        f"Testing rows:                     {test_rows:,}",
        f"Raw feature count:                {X_train_raw.shape[1]} (39 numerical, 3 categorical)",
        f"Transformed features:             {len(feature_names)}",
        "",
        "1. BINARY THREAT CLASSIFIER",
        "-" * 85,
        f"Model Architecture:               HistGradientBoostingClassifier",
        f"Target:                           label (0 = Normal, 1 = Attack)",
        f"Class Weighting:                  balanced",
        f"Accuracy:                         {binary_metrics['accuracy'] * 100:.2f}%",
        f"Precision:                        {binary_metrics['precision'] * 100:.2f}%",
        f"Recall:                           {binary_metrics['recall'] * 100:.2f}%",
        f"F1-Score:                         {binary_metrics['f1_score']:.4f}",
        f"ROC-AUC:                          {binary_metrics['roc_auc']:.4f}",
        f"Confusion Matrix (TN, FP, FN, TP): [{binary_metrics['confusion_matrix']['true_negative']:,}, {binary_metrics['confusion_matrix']['false_positive']:,}, {binary_metrics['confusion_matrix']['false_negative']:,}, {binary_metrics['confusion_matrix']['true_positive']:,}]",
        f"Model Artifact:                   {config.SUSPICIOUS_CLASSIFIER_PATH}",
        "",
        "2. MULTICLASS ATTACK-TYPE CLASSIFIER",
        "-" * 85,
        f"Model Architecture:               HistGradientBoostingClassifier",
        f"Target Taxonomy:                  attack_cat (excluding Normal)",
        f"Number of Attack Families:        {len(attack_classes)} ({attack_classes})",
        f"Accuracy on Attack Test Set:      {attack_metrics['accuracy'] * 100:.2f}%",
        f"Macro Precision:                  {attack_metrics['macro_precision'] * 100:.2f}%",
        f"Macro Recall:                     {attack_metrics['macro_recall'] * 100:.2f}%",
        f"Macro F1-Score:                   {attack_metrics['macro_f1']:.4f}",
        f"Weighted F1-Score:                {attack_metrics['weighted_f1']:.4f}",
        f"Model Artifact:                   {config.ATTACK_CLASSIFIER_PATH}",
        "",
        "3. ANOMALY DETECTION ENGINE",
        "-" * 85,
        f"Model Architecture:               IsolationForest",
        f"Training Profile:                 Trained solely on benign (Normal) network traffic",
        f"Contamination Configuration:      {config.ISOLATION_FOREST_CONTAMINATION}",
        f"Anomaly Recall (Attack Catch):    {anomaly_metrics['anomaly_recall_attack_detection_rate'] * 100:.2f}%",
        f"Anomaly Precision:                {anomaly_metrics['anomaly_precision'] * 100:.2f}%",
        f"Model Artifact:                   {config.ANOMALY_DETECTOR_PATH}",
        "",
        "4. EXPLAINABILITY",
        "-" * 85,
        "Framework:                        Scikit-Learn Native Permutation Importance",
        "Offline Status:                   SHAP package unavailable in offline environment; native permutation F1 used.",
        f"Top Feature Importance Plot:      {config.BINARY_FEATURE_IMPORTANCE_PATH}",
        "",
        "5. KNOWN LIMITATIONS",
        "-" * 85,
        "  1. Highly imbalanced rare attacks: Classes with very low sample counts (e.g. Worms with only 44 test samples)",
        "     exhibit lower macro recall compared to dominant classes like Generic and Exploits.",
        "  2. Non-IID network temporal shifts: Synthetically split lab datasets like UNSW-NB15 may suffer distribution",
        "     shift when applied to live production enterprise environments without domain adaptation.",
        "  3. Note on Production Readiness: These models are trained for the ThreatLens AI hackathon demonstration",
        "     pipeline and should undergo real-world canary calibration before critical perimeter deployment.",
        "=" * 85,
    ]

    with open(config.TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"      Full training summary written to: {config.TRAINING_SUMMARY_PATH}")
    print("=" * 85)
    print("ThreatLens AI ML Detection Pipeline Training Complete!")
    print("=" * 85)

    return {
        "binary_metrics": binary_metrics,
        "attack_metrics": attack_metrics,
        "anomaly_metrics": anomaly_metrics,
        "summary_file": config.TRAINING_SUMMARY_PATH,
    }


if __name__ == "__main__":
    run_full_training_pipeline()
