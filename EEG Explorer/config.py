from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

PROJECT_ROOT=Path(__file__).parent
DATA_DIR=PROJECT_ROOT/'data'
OUTPUT_DIR=PROJECT_ROOT/'output'
FIGURE_DIR=OUTPUT_DIR/'figures'
REPORT_DIR=OUTPUT_DIR/'reports'
MODEL_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
COMPARE_DIR = OUTPUT_DIR / "comparison"
PREDICTION_DIR = OUTPUT_DIR / "prediction"
DEFAULT_MODEL="svm_v1"
DEFAULT_MODEL_TYPE="svm"
MODEL_LIST = [
    ("svm_v2", "svm"),
    ("rf_v2", "rf"),
    ("lr_v1", "lr")
]


DIRS=[PROJECT_ROOT,DATA_DIR,OUTPUT_DIR,FIGURE_DIR,REPORT_DIR,MODEL_DIR,COMPARE_DIR,PREDICTION_DIR,EVALUATION_DIR]
for path in DIRS:
    path.mkdir(parents=True,exist_ok=True)

EEGLZ=DATA_DIR/'test_eeg.csv'

bands={
"delta":(1,4),
"theta":(4,8),
"alpha":(8,13),
"beta":(13,30),
"gamma":(30,50)
}



SAMPLING_RATE=160

FILTER_ORDER = 4
LOWCUT=1
HIGHCUT=40

NOTCH_FREQ=50
NOTCH_Q=30

window_size=1600
