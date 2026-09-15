from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Detect data directory (case-insensitive fallback)
def get_raw_data_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "Data" / "raw",
        PROJECT_ROOT / "DATA" / "RAW",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR = get_raw_data_dir()
TRAIN_RAW_FILE = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_RAW_FILE = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"

# Artifact and model output paths
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.json"

# Model paths
SUSPICIOUS_CLASSIFIER_PATH = MODELS_DIR / "suspicious_classifier.pkl"
ATTACK_CLASSIFIER_PATH = MODELS_DIR / "attack_classifier.pkl"
ATTACK_CLASSES_PATH = MODELS_DIR / "attack_classes.json"
ANOMALY_DETECTOR_PATH = MODELS_DIR / "anomaly_detector.pkl"

# Evaluation and report paths
PREPROCESSING_REPORT_PATH = OUTPUTS_DIR / "preprocessing_report.txt"
BINARY_METRICS_PATH = OUTPUTS_DIR / "binary_metrics.json"
BINARY_REPORT_PATH = OUTPUTS_DIR / "binary_classification_report.txt"
BINARY_CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "binary_confusion_matrix.png"
BINARY_FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "binary_feature_importance.png"

ATTACK_METRICS_PATH = OUTPUTS_DIR / "attack_metrics.json"
ATTACK_REPORT_PATH = OUTPUTS_DIR / "attack_classification_report.txt"
ATTACK_CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "attack_confusion_matrix.png"
TRAINING_SUMMARY_PATH = OUTPUTS_DIR / "ml_training_summary.txt"

# Target definitions
TARGET_BINARY = "label"
TARGET_MULTICLASS = "attack_cat"
IDENTIFIER_COLUMNS = ["id"]
DROP_COLUMNS = IDENTIFIER_COLUMNS + [TARGET_BINARY, TARGET_MULTICLASS]

# Feature definitions (42 features total)
NUMERICAL_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports"
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Preprocessing hyper-parameters and settings
NUMERICAL_IMPUTER_STRATEGY = "median"
CATEGORICAL_HANDLE_UNKNOWN = "ignore"
APPLY_NUMERICAL_SCALING = False

# Model Training Parameters
RANDOM_STATE = 42

# HistGradientBoostingClassifier settings
HGB_MAX_ITER = 150
HGB_LEARNING_RATE = 0.1
HGB_MAX_LEAF_NODES = 31

# Isolation Forest settings
ISOLATION_FOREST_N_ESTIMATORS = 100
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_MAX_SAMPLES = 256
