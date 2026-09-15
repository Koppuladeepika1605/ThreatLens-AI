"""
ThreatLens AI - Data Preprocessing Pipeline for UNSW-NB15
==========================================================

DATA LEAKAGE PREVENTION MANIFESTO:
----------------------------------
1. Strict Split Isolation:
   The official UNSW-NB15 training and testing splits are kept strictly separate.
   They are never merged or pooled prior to transformation.

2. Fit on Training ONLY:
   All transformers (numerical median imputer, categorical one-hot encoder, and
   optional scalers) are fitted EXCLUSIVELY on X_train.
   No statistics (mean, median, quantiles, or category dictionaries) are learned
   from X_test.

3. Out-of-Vocabulary Resilience:
   Categorical encoders use `handle_unknown="ignore"`. Any novel categories
   encountered in the test set (such as unseen connection states 'ACC' and 'CLO')
   are encoded as all-zeros without causing pipeline failure or data snooping.

4. Target & Identifier Separation:
   Non-feature columns ('id', 'label', 'attack_cat') are strictly isolated and
   removed prior to feature pipeline ingestion.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder

# Import centralized configuration
from src import config


def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Load raw CSV training and testing datasets from data/raw/ and remove
    the single corrupted/truncated training row.
    
    Returns:
        df_train (pd.DataFrame): Cleaned training dataset.
        df_test (pd.DataFrame): Cleaned testing dataset.
        orig_train_rows (int): Original row count before dropping corrupt record.
    """
    if not config.TRAIN_RAW_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {config.TRAIN_RAW_FILE}")
    if not config.TEST_RAW_FILE.exists():
        raise FileNotFoundError(f"Testing file not found: {config.TEST_RAW_FILE}")

    # Load raw CSVs (original CSV files remain untouched)
    df_train = pd.read_csv(config.TRAIN_RAW_FILE)
    df_test = pd.read_csv(config.TEST_RAW_FILE)

    orig_train_rows = len(df_train)

    # Clean and standardize column names (strip accidental whitespace)
    df_train.columns = df_train.columns.astype(str).str.strip()
    df_test.columns = df_test.columns.astype(str).str.strip()

    # Drop the verified truncated/corrupted row in training set (id == 173436 or where targets are null)
    # This row has 18 missing values due to line truncation in the raw file.
    corrupt_mask = (
        df_train["id"].isin([173436]) |
        df_train[config.TARGET_BINARY].isna() |
        df_train[config.TARGET_MULTICLASS].isna()
    )
    df_train_cleaned = df_train[~corrupt_mask].copy().reset_index(drop=True)

    return df_train_cleaned, df_test.copy().reset_index(drop=True), orig_train_rows


def prepare_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Separate features from target variables, strip whitespace from strings,
    replace infinities with NaN for safe imputation, and ensure correct data types.
    
    Returns:
        X_train, y_train_binary, y_train_attack,
        X_test, y_test_binary, y_test_attack
    """
    # 1. Target separation
    y_train_binary = df_train[config.TARGET_BINARY].astype(int)
    y_train_attack = df_train[config.TARGET_MULTICLASS].astype(str).str.strip()

    y_test_binary = df_test[config.TARGET_BINARY].astype(int)
    y_test_attack = df_test[config.TARGET_MULTICLASS].astype(str).str.strip()

    # 2. Extract ML features by dropping id, label, and attack_cat
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df_train.columns]
    X_train = df_train.drop(columns=cols_to_drop).copy()
    X_test = df_test.drop(columns=cols_to_drop).copy()

    # 3. Clean categorical strings (strip whitespace, retain '-' as legitimate category)
    for col in config.CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str).str.strip()
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).str.strip()

    # 4. Convert numerical features to float and replace infinities with NaN
    for col in config.NUMERICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    return (
        X_train,
        y_train_binary,
        y_train_attack,
        X_test,
        y_test_binary,
        y_test_attack,
    )


def fit_preprocessor(
    X_train: pd.DataFrame,
    apply_scaling: bool = config.APPLY_NUMERICAL_SCALING
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build and fit the scikit-learn ColumnTransformer EXCLUSIVELY on X_train.
    
    Data Leakage Protection:
    - Never pass X_test or pooled data into this function.
    - Fits SimpleImputer(strategy='median') on X_train numerical columns.
    - Fits OneHotEncoder(handle_unknown='ignore') on X_train categorical columns.
    
    Returns:
        preprocessor (ColumnTransformer): Fitted pipeline.
        feature_names (List[str]): List of transformed feature names.
    """
    # Numerical pipeline: Median imputation (preserves tree split geometry)
    num_steps = [
        ("imputer", SimpleImputer(strategy=config.NUMERICAL_IMPUTER_STRATEGY))
    ]
    if apply_scaling:
        num_steps.append(("scaler", RobustScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    # Categorical pipeline: Missing value fill + OneHotEncoder with handle_unknown='ignore'
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=config.CATEGORICAL_HANDLE_UNKNOWN,
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, config.NUMERICAL_FEATURES),
            ("cat", cat_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Fit strictly on X_train
    preprocessor.fit(X_train)

    # Extract feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    return preprocessor, feature_names


def transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform training and testing features using the pre-fitted preprocessor.
    Notice: .transform() is used on both sets; .fit() is never called here.
    
    Returns:
        X_train_transformed (np.ndarray): Transformed training feature matrix.
        X_test_transformed (np.ndarray): Transformed testing feature matrix.
    """
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed


def save_preprocessing_artifacts(
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    label_encoder: LabelEncoder,
    save_dir: Path = config.MODELS_DIR,
) -> Dict[str, Path]:
    """
    Save the fitted preprocessor pipeline, feature names catalog, and label encoder metadata.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save preprocessor.pkl
    preprocessor_file = save_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_file)

    # 2. Save feature_names.json
    feature_names_file = save_dir / "feature_names.json"
    with open(feature_names_file, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Save label_encoder.json metadata for multiclass attack taxonomy
    label_encoder_file = save_dir / "label_encoder.json"
    class_mapping = {
        cls_name: int(idx)
        for idx, cls_name in enumerate(label_encoder.classes_)
    }
    index_mapping = {
        int(idx): cls_name
        for idx, cls_name in enumerate(label_encoder.classes_)
    }
    metadata = {
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "class_to_index": class_mapping,
        "index_to_class": index_mapping,
    }
    with open(label_encoder_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "preprocessor": preprocessor_file,
        "feature_names": feature_names_file,
        "label_encoder": label_encoder_file,
    }


def validate_pipeline(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_train_tf: np.ndarray,
    X_test_tf: np.ndarray,
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    y_train_attack: pd.Series,
    y_test_attack: pd.Series,
    feature_names: List[str],
) -> List[str]:
    """
    Execute rigorous automated integrity checks on preprocessed artifacts.
    """
    checks_passed = []

    # 1. Check for NaNs
    train_nans = int(np.isnan(X_train_tf).sum())
    test_nans = int(np.isnan(X_test_tf).sum())
    assert train_nans == 0, f"Validation failure: X_train contains {train_nans} NaNs"
    assert test_nans == 0, f"Validation failure: X_test contains {test_nans} NaNs"
    checks_passed.append("Transformed matrices have zero NaNs (train=0, test=0).")

    # 2. Transformed feature counts match
    assert X_train_tf.shape[1] == X_test_tf.shape[1], (
        f"Mismatch: Train features {X_train_tf.shape[1]} vs Test features {X_test_tf.shape[1]}"
    )
    assert X_train_tf.shape[1] == len(feature_names), (
        f"Mismatch: Matrix columns {X_train_tf.shape[1]} vs Feature names {len(feature_names)}"
    )
    checks_passed.append(
        f"Feature dimensionality perfectly matches across train, test, and feature_names catalog ({X_train_tf.shape[1]} columns)."
    )

    # 3. Target lengths match feature row counts
    assert len(y_train_binary) == X_train_tf.shape[0], "Train binary target row count mismatch"
    assert len(y_train_attack) == X_train_tf.shape[0], "Train multiclass target row count mismatch"
    assert len(y_test_binary) == X_test_tf.shape[0], "Test binary target row count mismatch"
    assert len(y_test_attack) == X_test_tf.shape[0], "Test multiclass target row count mismatch"
    checks_passed.append("Row counts for X and target vectors match identically.")

    # 4. Zero target or identifier leakage
    for forbidden in config.DROP_COLUMNS:
        assert forbidden not in X_train_raw.columns, f"Leakage: {forbidden} found in X_train"
        assert forbidden not in X_test_raw.columns, f"Leakage: {forbidden} found in X_test"
    checks_passed.append("Leakage verification: 'id', 'label', and 'attack_cat' successfully absent from feature matrices.")

    return checks_passed


def generate_report(
    orig_train_rows: int,
    cleaned_train_rows: int,
    test_rows: int,
    raw_feature_count: int,
    transformed_feature_count: int,
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    y_train_attack: pd.Series,
    y_test_attack: pd.Series,
    validation_checks: List[str],
    saved_artifacts: Dict[str, Path],
) -> Path:
    """
    Generate and save comprehensive preprocessing report to outputs/preprocessing_report.txt.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = config.PREPROCESSING_REPORT_PATH

    lines = []
    lines.append("=" * 85)
    lines.append("           THREATLENS AI - DATA PREPROCESSING & PIPELINE REPORT")
    lines.append("=" * 85)
    lines.append(f"Source Training File:     {config.TRAIN_RAW_FILE}")
    lines.append(f"Source Testing File:      {config.TEST_RAW_FILE}")
    lines.append(f"Original Training Rows:   {orig_train_rows:,}")
    lines.append(f"Corrupt/Truncated Dropped: 1 (id=173436, truncated at column 27)")
    lines.append(f"Cleaned Training Rows:    {cleaned_train_rows:,}")
    lines.append(f"Testing Rows:             {test_rows:,}")
    lines.append(f"Total Operational Rows:   {cleaned_train_rows + test_rows:,}")
    lines.append("-" * 85)
    lines.append("")

    lines.append("1. FEATURE CATALOG & TRANSFORMED DIMENSIONS")
    lines.append("-" * 85)
    lines.append(f"Original Columns in CSV:          45")
    lines.append(f"Excluded Non-Features:            3 ('id', 'label', 'attack_cat')")
    lines.append(f"Raw Input ML Features:            {raw_feature_count}")
    lines.append(f"  - Numerical Features Count:     {len(config.NUMERICAL_FEATURES)}")
    lines.append(f"  - Categorical Features Count:   {len(config.CATEGORICAL_FEATURES)} ({config.CATEGORICAL_FEATURES})")
    lines.append(f"Final Transformed Feature Count:  {transformed_feature_count}")
    lines.append("")

    lines.append("2. TARGET VARIABLE DISTRIBUTIONS")
    lines.append("-" * 85)
    lines.append("A. Binary Target ('label'):")
    tr_norm = int((y_train_binary == 0).sum())
    tr_att = int((y_train_binary == 1).sum())
    te_norm = int((y_test_binary == 0).sum())
    te_att = int((y_test_binary == 1).sum())

    lines.append(f"  - Training Set:")
    lines.append(f"      * Normal (0): {tr_norm:>8,} ({tr_norm/cleaned_train_rows*100:>5.2f}%)")
    lines.append(f"      * Attack (1): {tr_att:>8,} ({tr_att/cleaned_train_rows*100:>5.2f}%)")
    lines.append(f"  - Testing Set:")
    lines.append(f"      * Normal (0): {te_norm:>8,} ({te_norm/test_rows*100:>5.2f}%)")
    lines.append(f"      * Attack (1): {te_att:>8,} ({te_att/test_rows*100:>5.2f}%)")
    lines.append("")

    lines.append("B. Multiclass Attack Taxonomy ('attack_cat'):")
    all_cats = sorted(list(set(y_train_attack.unique()).union(set(y_test_attack.unique()))))
    lines.append(f"  {'Category':<18} | {'Train Count':<14} | {'Train %':<9} | {'Test Count':<14} | {'Test %':<9}")
    lines.append("  " + "-" * 70)
    for cat in all_cats:
        tc = int((y_train_attack == cat).sum())
        sc = int((y_test_attack == cat).sum())
        lines.append(f"  {cat:<18} | {tc:>14,} | {tc/cleaned_train_rows*100:>8.2f}% | {sc:>14,} | {sc/test_rows*100:>8.2f}%")
    lines.append("")

    lines.append("3. DATA LEAKAGE PREVENTION & PIPELINE CONFIGURATION")
    lines.append("-" * 85)
    lines.append("  * Fit Isolation: ColumnTransformer fitted exclusively on X_train.")
    lines.append("  * Test Set Integrity: X_test strictly transformed via .transform(). No test-set fitting occurred.")
    lines.append("  * Numerical Pipeline: SimpleImputer(strategy='median') learned medians from X_train only.")
    lines.append("  * Categorical Pipeline: OneHotEncoder(handle_unknown='ignore', sparse_output=False).")
    lines.append("    - Service '-' preserved as valid category 'no application service'.")
    lines.append("    - Novel test categories (e.g. states 'ACC', 'CLO') cleanly map to zeros without error.")
    lines.append("  * Scaling Status: Optional scaling = False. Raw tree split boundaries preserved for XGBoost.")
    lines.append("  * Resampling Status: SMOTE = False. No synthetic distortion applied to training or testing distributions.")
    lines.append("")

    lines.append("4. AUTOMATED VALIDATION CHECKS")
    lines.append("-" * 85)
    for check in validation_checks:
        lines.append(f"  [PASS] {check}")
    lines.append("")

    lines.append("5. SAVED ARTIFACTS")
    lines.append("-" * 85)
    for name, path in saved_artifacts.items():
        lines.append(f"  - {name:<15}: {path}")
    lines.append("=" * 85)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report_file


def preprocess_dataset() -> Dict[str, Any]:
    """
    Execute end-to-end preprocessing workflow:
    1. Load and clean raw datasets.
    2. Prepare features and target vectors.
    3. Fit preprocessing ColumnTransformer strictly on X_train.
    4. Transform X_train and X_test.
    5. Fit label encoder on y_train_attack.
    6. Save pipeline, feature names, and metadata artifacts.
    7. Execute automated validation tests.
    8. Write comprehensive preprocessing report.
    
    Returns:
        dict containing preprocessed matrices, targets, and artifact filepaths.
    """
    print("=" * 80)
    print("ThreatLens AI - Running UNSW-NB15 Preprocessing Pipeline")
    print("=" * 80)

    # Step 1: Load and clean data
    print("[1/6] Loading raw data and cleaning truncated row...")
    df_train, df_test, orig_train_rows = load_and_clean_data()
    print(f"      Cleaned train rows: {len(df_train):,}, Cleaned test rows: {len(df_test):,}")

    # Step 2: Prepare features and targets
    print("[2/6] Separating features, targets, and sanitizing types...")
    (
        X_train,
        y_train_binary,
        y_train_attack,
        X_test,
        y_test_binary,
        y_test_attack,
    ) = prepare_features(df_train, df_test)
    raw_feature_count = X_train.shape[1]
    print(f"      Raw ML features: {raw_feature_count} (39 numerical, 3 categorical)")

    # Step 3: Fit preprocessor strictly on X_train (Leakage Prevention)
    print("[3/6] Fitting ColumnTransformer exclusively on X_train...")
    preprocessor, feature_names = fit_preprocessor(X_train)
    print(f"      Transformed feature space: {len(feature_names)} features")

    # Step 4: Transform X_train and X_test
    print("[4/6] Transforming feature matrices (using fitted preprocessor)...")
    X_train_tf, X_test_tf = transform_features(preprocessor, X_train, X_test)

    # Step 5: Encode multiclass target labels
    print("[5/6] Encoding multiclass attack categories...")
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_attack)
    y_train_attack_encoded = label_encoder.transform(y_train_attack)
    y_test_attack_encoded = label_encoder.transform(y_test_attack)

    # Save artifacts to models/
    saved_artifacts = save_preprocessing_artifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_encoder=label_encoder,
    )
    print(f"      Artifacts saved to {config.MODELS_DIR}")

    # Step 6: Validate pipeline integrity
    print("[6/6] Running validation checks and generating report...")
    validation_checks = validate_pipeline(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train_tf=X_train_tf,
        X_test_tf=X_test_tf,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        y_train_attack=y_train_attack,
        y_test_attack=y_test_attack,
        feature_names=feature_names,
    )
    for v in validation_checks:
        print(f"      [PASS] {v}")

    # Write report to outputs/preprocessing_report.txt
    report_file = generate_report(
        orig_train_rows=orig_train_rows,
        cleaned_train_rows=len(df_train),
        test_rows=len(df_test),
        raw_feature_count=raw_feature_count,
        transformed_feature_count=len(feature_names),
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        y_train_attack=y_train_attack,
        y_test_attack=y_test_attack,
        validation_checks=validation_checks,
        saved_artifacts=saved_artifacts,
    )
    print(f"Preprocessing completed successfully! Report generated: {report_file}")
    print("=" * 80)

    return {
        "X_train_transformed": X_train_tf,
        "X_test_transformed": X_test_tf,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary,
        "y_train_attack": y_train_attack,
        "y_test_attack": y_test_attack,
        "y_train_attack_encoded": y_train_attack_encoded,
        "y_test_attack_encoded": y_test_attack_encoded,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "report_file": report_file,
    }


if __name__ == "__main__":
    preprocess_dataset()
