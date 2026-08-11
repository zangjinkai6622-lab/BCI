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


SAMPLING_RATE=250

FILTER_ORDER = 4
LOWCUT=1
HIGHCUT=40

NOTCH_FREQ=50
NOTCH_Q=30

window_size=1600


# 窗口切分策略 - 按数据集显式配置
# 可选策略: "annotation" | "fixed_offset"
#   "annotation"  : 使用 raw.annotations.duration 作为窗口长度（旧 EDF 数据）
#   "fixed_offset": 相对 onset 固定偏移窗口（BCI IV 2a 等事件点式数据集）
WINDOW_STRATEGY = "fixed_offset"

# annotation 模式（旧数据用）
ANNOT_MIN_DURATION_SEC = 0.5
ANNOT_PAD_START_SEC = 0.0
ANNOT_PAD_END_SEC = 0.0

# fixed_offset 模式（BCI IV 2a 用，标准 MI 范式：提示音后 0.5s~4.5s 共 4s）
FIXED_WINDOW_TMIN = 0.5
FIXED_WINDOW_TMAX = 4.5

# 通用参数（两种模式都用）
MIN_WINDOW_SAMPLES = 50

# 数据集事件过滤 & 标签映射
# BCI IV 2a 等数据集的非 trial 事件码：切窗口时直接跳过（不生成训练样本）
SKIP_EVENT_CODES = {
    32766,   # Boundary / Comment 边界标记（GDF特有）
    276,     # Fixation cross 十字注视点提示
    277,     # Feedback start 反馈开始
    1072,    # Eye movements / artifact rejected 眼动或伪迹拒绝
    1023,    # Rejected trial 被拒绝的试次
    772,     # Tongue 舌头（项目未定义该类别，按方案 A 直接跳过）
}

# BCI IV 2a 标签事件码 → 项目统一标签整数（与 label.LABEL_NAME 对齐）
# 当前保留三分类 + rest：769左手 / 770右手 / 771双脚（772 舌头已加入 SKIP_EVENT_CODES 跳过）
BCI2A_EVENT_TO_LABEL = {
    769: 1,   # left_hand
    770: 2,   # right_hand
    771: 4,   # both_feet
}

# 无标签事件码（BCI IV 2a 评估集 E 文件用，训练阶段应跳过）
BCI2A_UNLABELED_CODES = {768, 783}
