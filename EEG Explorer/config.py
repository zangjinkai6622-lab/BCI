from pathlib import Path

PROJECT_ROOT=Path(__file__).parent

DATA_DIR=PROJECT_ROOT/'data'

OUTPUT_DIR=PROJECT_ROOT/'output'

FIGURE_DIR=OUTPUT_DIR/'figures'

REPORT_DIR=OUTPUT_DIR/'reports'

DIRS=[PROJECT_ROOT,DATA_DIR,OUTPUT_DIR,FIGURE_DIR,REPORT_DIR]
for path in DIRS:
    path.mkdir(parents=True,exist_ok=True)

EEGLZ=DATA_DIR/'test_eeg.csv'