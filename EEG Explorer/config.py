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

bands={
"delta":(1,4),
"theta":(4,8),
"alpha":(8,13),
"beta":(13,30),
"gamma":(30,50)
}



SAMPLING_RATE=100

FILTER_ORDER = 4
LOWCUT=1
HIGHCUT=40

NOTCH_FREQ=50
NOTCH_Q=30

window_size=1600
