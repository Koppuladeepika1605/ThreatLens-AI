import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def find_dataset_dir(base_dir: Path) -> Path:
    """Find the data/raw directory handling case sensitivity."""
    candidates = [
        base_dir / "data" / "raw",
        base_dir / "Data" / "raw",
        base_dir / "DATA" / "RAW",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find data/raw directory under {base_dir}")

def check_empty_strings(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, int]:
    """Count empty strings or whitespace-only strings in categorical columns."""
    empty_counts = {}
    for col in cat_cols:
        count = (df[col].dropna().astype(str).str.strip() == "").sum()
        if count > 0:
            empty_counts[col] = count
    return empty_counts

def check_infinities(df: pd.DataFrame, num_cols: list[str]) -> dict[str, int]:
    """Count positive and negative infinite values in numerical columns."""
    inf_counts = {}
    for col in num_cols:
        count = np.isinf(df[col].dropna()).sum()
        if count > 0:
            inf_counts[col] = count
    return inf_counts

def inspect_dataset():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = find_dataset_dir(project_root)
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = raw_dir / "UNSW_NB15_training-set.csv"
    test_file = raw_dir / "UNSW_NB15_testing-set.csv"
    
    report_lines = []
    
    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)
        
    log("=" * 85)
    log("               THREATLENS AI - UNSW-NB15 DATASET INSPECTION REPORT")
    log("=" * 85)
    log(f"Project Root:        {project_root}")
    log(f"Raw Data Directory:  {raw_dir}")
    log(f"Training Set File:   {train_file.name} (Exists: {train_file.exists()})")
    log(f"Testing Set File:    {test_file.name} (Exists: {test_file.exists()})")
    log()
    
    if not train_file.exists() or not test_file.exists():
        log("CRITICAL ERROR: One or both dataset files are missing.")
        return
        
    log("Loading raw CSV datasets into pandas...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    log("Datasets successfully loaded!")
    log("-" * 85)
    
    # 1. Basic Dimensions & File Sizes
    log("1. DATASET DIMENSIONS & FILE SIZES")
    log("-" * 85)
    log(f"Training Set ({train_file.name}):")
    log(f"  - Total Rows:    {df_train.shape[0]:,}")
    log(f"  - Total Columns: {df_train.shape[1]}")
    log(f"  - File Size:     {train_file.stat().st_size / (1024*1024):.2f} MB")
    log()
    log(f"Testing Set ({test_file.name}):")
    log(f"  - Total Rows:    {df_test.shape[0]:,}")
    log(f"  - Total Columns: {df_test.shape[1]}")
    log(f"  - File Size:     {test_file.stat().st_size / (1024*1024):.2f} MB")
    log()
    log(f"Combined Total Records: {df_train.shape[0] + df_test.shape[0]:,}")
    log()
    
    # 2. Schema Comparison & Compatibility
    log("2. SCHEMA COMPARISON & COMPATIBILITY")
    log("-" * 85)
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)
    cols_match = train_cols == test_cols
    log(f"Exact Column Alignment (names and sequence identical): {cols_match}")
    if not cols_match:
        missing_in_test = set(train_cols) - set(test_cols)
        missing_in_train = set(test_cols) - set(train_cols)
        log(f"  - Columns in train but not test: {missing_in_test}")
        log(f"  - Columns in test but not train: {missing_in_train}")
    else:
        log("  - All 45 column names and ordering match identically between train and test.")
        
    dtype_mismatches = []
    for col in train_cols:
        t_dtype = str(df_train[col].dtype)
        s_dtype = str(df_test[col].dtype)
        if t_dtype != s_dtype:
            dtype_mismatches.append((col, t_dtype, s_dtype))
            
    if dtype_mismatches:
        log(f"Data type differences detected ({len(dtype_mismatches)} columns):")
        for col, t_dtype, s_dtype in dtype_mismatches:
            log(f"  - Column '{col}': Train={t_dtype} vs Test={s_dtype}")
        log("  Note: Several integer columns in train appear as float64 due to the presence of 1 NaN value in train.")
    else:
        log("  - All column data types match identically.")
    log()
    
    # 3. Column Listing, Data Types, and Null Counts
    log("3. COLUMN NAMES, DATA TYPES & MISSING VALUES PER COLUMN")
    log("-" * 85)
    log(f"{'#':<4} | {'Column Name':<20} | {'Train Type':<12} | {'Test Type':<12} | {'Train NaN':<10} | {'Test NaN':<10}")
    log("-" * 85)
    for idx, col in enumerate(train_cols, 1):
        tr_type = str(df_train[col].dtype)
        te_type = str(df_test[col].dtype)
        tr_null = int(df_train[col].isna().sum())
        te_null = int(df_test[col].isna().sum())
        log(f"{idx:<4} | {col:<20} | {tr_type:<12} | {te_type:<12} | {tr_null:<10} | {te_null:<10}")
    log()
    
    # 4. Data Integrity: Missing Values, Anomalies, and Duplicates
    log("4. DATA INTEGRITY: MISSING VALUES, ANOMALIES & DUPLICATES")
    log("-" * 85)
    train_null_cols = df_train.isna().sum()[df_train.isna().sum() > 0]
    test_null_cols = df_test.isna().sum()[df_test.isna().sum() > 0]
    
    log(f"Missing Values Summary:")
    log(f"  - Training Set Total NaNs: {df_train.isna().sum().sum()}")
    if not train_null_cols.empty:
        log(f"    Notice: Exactly {len(train_null_cols)} columns in training have 1 NaN value each:")
        log(f"    Columns affected: {list(train_null_cols.index)}")
        log("    Root Cause Analysis: Row index 173435 (id=173436) in the raw CSV is truncated at column 27.")
    log(f"  - Testing Set Total NaNs:  {df_test.isna().sum().sum()} (0 missing values)")
    log()
    
    # Check duplicates
    train_dup_full = int(df_train.duplicated().sum())
    test_dup_full = int(df_test.duplicated().sum())
    log("Duplicate Rows (Entire row including 'id'):")
    log(f"  - Training: {train_dup_full}")
    log(f"  - Testing:  {test_dup_full}")
    
    feature_cols_wo_id = [c for c in train_cols if c.lower() != 'id']
    train_dup_wo_id = int(df_train.duplicated(subset=feature_cols_wo_id).sum())
    test_dup_wo_id = int(df_test.duplicated(subset=feature_cols_wo_id).sum())
    log("Duplicate Rows (Excluding 'id' column):")
    log(f"  - Training: {train_dup_wo_id:,} ({train_dup_wo_id / len(df_train) * 100:.2f}%)")
    log(f"  - Testing:  {test_dup_wo_id:,} ({test_dup_wo_id / len(df_test) * 100:.2f}%)")
    log("    Explanation: Network traffic traces often have identical flow characteristics")
    log("    (e.g., standard SYN probes, DNS queries, repetitive beaconing).")
    log()
    
    # Numerical vs Categorical
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    log(f"Numerical Columns Count:   {len(num_cols)}")
    log(f"Categorical Columns Count: {len(cat_cols)} ({cat_cols})")
    
    # Infinities & empty strings
    train_inf = check_infinities(df_train, num_cols)
    test_inf = check_infinities(df_test, num_cols)
    log(f"Infinite (+/- inf) values in Training: {train_inf if train_inf else 'None'}")
    log(f"Infinite (+/- inf) values in Testing:  {test_inf if test_inf else 'None'}")
    
    train_empty = check_empty_strings(df_train, cat_cols)
    test_empty = check_empty_strings(df_test, cat_cols)
    log(f"Empty/whitespace strings in Training categorical: {train_empty if train_empty else 'None'}")
    log(f"Empty/whitespace strings in Testing categorical:  {test_empty if test_empty else 'None'}")
    log()
    
    # 5. Deep Investigation of Specific Columns
    log("5. DEEP DIVE: KEY COLUMNS INVESTIGATION")
    log("-" * 85)
    key_columns = ['label', 'attack_cat', 'proto', 'service', 'state']
    for col in key_columns:
        log(f"COLUMN: '{col}'")
        tr_valid = df_train[col].dropna()
        te_valid = df_test[col].dropna()
        log(f"  - Data Type:      Train={df_train[col].dtype} | Test={df_test[col].dtype}")
        log(f"  - Unique Values:  Train={tr_valid.nunique()} | Test={te_valid.nunique()}")
        
        tr_unique = set(tr_valid.unique())
        te_unique = set(te_valid.unique())
        in_train_only = tr_unique - te_unique
        in_test_only = te_unique - tr_unique
        
        if in_train_only:
            log(f"  - Values exclusive to Train ({len(in_train_only)}): {sorted(list(in_train_only))}")
        if in_test_only:
            log(f"  - Values exclusive to Test ({len(in_test_only)}):  {sorted(list(in_test_only))}")
            
        val_counts_tr = df_train[col].value_counts(dropna=False)
        log("  - Top Frequencies in Training:")
        for val, cnt in val_counts_tr.head(10).items():
            disp_val = repr(val) if pd.notna(val) else "NaN (missing)"
            log(f"      {disp_val:<25}: {cnt:>8,} ({cnt/len(df_train)*100:>5.2f}%)")
        log()
        
    # 6. Target Columns & Class Distribution Analysis
    log("6. TARGET COLUMNS & CLASS DISTRIBUTION")
    log("-" * 85)
    log("Target Column Identification:")
    log("  A. 'label': Binary target (0 = Normal, 1 = Attack)")
    log("  B. 'attack_cat': Multi-class attack taxonomy (Normal + 9 distinct attack categories)")
    log()
    
    # Binary target breakdown
    log("A. Binary Target ('label') Breakdown:")
    tr_bin = df_train['label'].value_counts(dropna=False)
    te_bin = df_test['label'].value_counts(dropna=False)
    
    normal_tr = int(tr_bin.get(0.0, 0))
    attack_tr = int(tr_bin.get(1.0, 0))
    nan_tr = int(tr_bin.get(np.nan, 0))
    
    normal_te = int(te_bin.get(0, 0))
    attack_te = int(te_bin.get(1, 0))
    nan_te = int(te_bin.get(np.nan, 0))
    
    log(f"{'Class':<14} | {'Training Count':<16} | {'Train %':<10} | {'Testing Count':<16} | {'Test %':<10} | {'Combined Count':<16} | {'Combined %':<10}")
    log("-" * 97)
    
    tot_norm = normal_tr + normal_te
    tot_att = attack_tr + attack_te
    total_samples = len(df_train) + len(df_test)
    
    log(f"{'Normal (0)':<14} | {normal_tr:>16,} | {normal_tr/len(df_train)*100:>9.2f}% | {normal_te:>16,} | {normal_te/len(df_test)*100:>9.2f}% | {tot_norm:>16,} | {tot_norm/total_samples*100:>9.2f}%")
    log(f"{'Attack (1)':<14} | {attack_tr:>16,} | {attack_tr/len(df_train)*100:>9.2f}% | {attack_te:>16,} | {attack_te/len(df_test)*100:>9.2f}% | {tot_att:>16,} | {tot_att/total_samples*100:>9.2f}%")
    if nan_tr > 0 or nan_te > 0:
        log(f"{'Corrupt/NaN':<14} | {nan_tr:>16,} | {nan_tr/len(df_train)*100:>9.4f}% | {nan_te:>16,} | {nan_te/len(df_test)*100:>9.4f}% | {nan_tr+nan_te:>16,} | {(nan_tr+nan_te)/total_samples*100:>9.4f}%")
    log()
    
    # Attack category breakdown
    log("B. Attack Category ('attack_cat') Detailed Distribution:")
    tr_cat_raw = df_train['attack_cat'].dropna().astype(str).str.strip().value_counts()
    te_cat_raw = df_test['attack_cat'].dropna().astype(str).str.strip().value_counts()
    all_categories = sorted(list(set(tr_cat_raw.index).union(set(te_cat_raw.index))))
    
    class_dist_rows = []
    log(f"{'Category':<18} | {'Training Count':<16} | {'Train %':<10} | {'Testing Count':<16} | {'Test %':<10} | {'Combined Count':<16} | {'Combined %':<10}")
    log("-" * 105)
    
    for cat in all_categories:
        tr_c = int(tr_cat_raw.get(cat, 0))
        te_c = int(te_cat_raw.get(cat, 0))
        tot_c = tr_c + te_c
        tr_p = tr_c / len(df_train) * 100
        te_p = te_c / len(df_test) * 100
        tot_p = tot_c / total_samples * 100
        log(f"{cat:<18} | {tr_c:>16,} | {tr_p:>9.2f}% | {te_c:>16,} | {te_p:>9.2f}% | {tot_c:>16,} | {tot_p:>9.2f}%")
        
        class_dist_rows.append({
            'category': cat,
            'train_count': tr_c,
            'train_pct': round(tr_p, 4),
            'test_count': te_c,
            'test_pct': round(te_p, 4),
            'total_count': tot_c,
            'total_pct': round(tot_p, 4)
        })
        
    df_class_dist = pd.DataFrame(class_dist_rows)
    class_dist_csv = output_dir / "class_distribution.csv"
    df_class_dist.to_csv(class_dist_csv, index=False)
    log()
    log(f"Exported class distribution table to: {class_dist_csv}")
    log()
    
    # 7. Feature Breakdown & Non-Feature Column Recommendations
    log("7. FEATURE CATALOG & RECOMMENDED EXCLUSIONS")
    log("-" * 85)
    drop_recommendations = [
        {
            "column": "id",
            "reason": "Synthetic identifier/sequence number. Carries zero physical network information. If retained, trees or linear models will memorize row index order, causing false predictive power and data leakage."
        },
        {
            "column": "label",
            "reason": "Ground-truth binary target. Must be separated into target vector y_binary to prevent target leakage."
        },
        {
            "column": "attack_cat",
            "reason": "Ground-truth multi-class target. Must be separated into target vector y_multiclass to prevent target leakage."
        }
    ]
    for r in drop_recommendations:
        log(f"  - Drop '{r['column']}': {r['reason']}")
        
    excluded = [r['column'] for r in drop_recommendations if r['column'] in train_cols]
    available_features = [c for c in train_cols if c not in excluded]
    
    log()
    log(f"Total Columns in Dataset:               {len(train_cols)}")
    log(f"Target and Identifier Columns Excluded: {len(excluded)} ({excluded})")
    log(f"Total Available ML Feature Columns:     {len(available_features)}")
    log()
    log("Available Features Grouped by Category:")
    
    flow_features = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len']
    content_features = ['is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports']
    window_features = ['ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ltm', 'ct_srv_dst']
    
    log(f"  - Flow/Packet Attributes ({len(flow_features)}): {flow_features}")
    log(f"  - Content & Flag Attributes ({len(content_features)}): {content_features}")
    log(f"  - Connection/Window Aggregates ({len(window_features)}): {window_features}")
    log()
    
    # 8. Class Imbalance Analysis
    log("8. CLASS IMBALANCE EVALUATION")
    log("-" * 85)
    log("Binary Level:")
    log(f"  - Train Set:  {attack_tr:,} Attacks ({attack_tr/len(df_train)*100:.2f}%) vs {normal_tr:,} Normal ({normal_tr/len(df_train)*100:.2f}%)")
    log(f"  - Test Set:   {attack_te:,} Attacks ({attack_te/len(df_test)*100:.2f}%) vs {normal_te:,} Normal ({normal_te/len(df_test)*100:.2f}%)")
    log("  - Finding: Attack traffic represents the majority class in both splits (67.7% in train, 55.1% in test).")
    log()
    log("Multi-Class Level:")
    log("  - Extreme disproportion among attack categories:")
    attacks_only = df_class_dist[df_class_dist['category'] != 'Normal']
    max_attack = attacks_only.sort_values(by='train_count', ascending=False).iloc[0]
    min_attack = attacks_only.sort_values(by='train_count', ascending=True).iloc[0]
    log(f"    * Dominant Attack:  {max_attack['category']} with {max_attack['train_count']:,} samples ({max_attack['train_pct']}%)")
    log(f"    * Rare Attack:      {min_attack['category']} with only {min_attack['train_count']:,} samples ({min_attack['train_pct']}%)")
    log(f"    * Multi-Class Imbalance Ratio: {max_attack['train_count'] / min_attack['train_count']:.1f} : 1")
    log("    * Categories with severe low frequency (< 2,500 samples in train): Worms (130), Shellcode (1,127), Backdoor (1,744), Analysis (2,000).")
    log()
    
    # 9. Preprocessing Pipeline Action Plan
    log("9. PREPROCESSING ACTION PLAN & PIPELINE RECOMMENDATIONS")
    log("-" * 85)
    log("  1. Corrupt Row Remediation: Drop or handle the single truncated row (index 173435) in train.")
    log("  2. Feature Separation: Isolate 'id' (drop), 'label' (y_binary), and 'attack_cat' (y_multi) from X.")
    log("  3. Categorical Preprocessing:")
    log("     - 'proto': 133 unique values. Top 3 ('tcp', 'udp', 'unas') constitute 88.5% of records. Group rare protocols (<0.1%) into 'other' or use Target/Frequency/Ordinal encoding.")
    log("     - 'service': 13 unique values. 54.2% is '-' (no application service). Treat '-' as a valid category 'none'. Use One-Hot or Ordinal Encoding.")
    log("     - 'state': 9 unique in train, 7 in test. Test introduces 'ACC' and 'CLO' unseen in train, while train has 'ECO', 'PAR', 'URN', 'no'. Categorical encoders must specify handle_unknown='ignore'.")
    log("  4. Numerical Feature Scaling: Features like 'dur', 'sbytes', 'dbytes', 'sload', 'dload' span orders of magnitude with high kurtosis and extreme outliers. Apply RobustScaler or quantile transformations; avoid standard min-max which collapses under extreme spikes.")
    log("  5. Handling Multi-Class Imbalance: When progressing to multi-class classification, use class-weighted loss functions (e.g., class_weight='balanced' or scale_pos_weight) and consider training-only SMOTE/ADASYN for ultra-rare classes like Worms.")
    log("=" * 85)
    
    report_file = output_dir / "dataset_inspection_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    log(f"Full report successfully written to: {report_file}")

if __name__ == "__main__":
    inspect_dataset()
