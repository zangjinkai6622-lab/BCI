"""AIC 比赛项目配置：路径 / 信号参数 / 模型超参 / 实验常量。

原则：所有可调常量集中在此文件，逻辑代码中不写死魔法数字。
信号处理参数与 EEG Explorer 练习项目保持一致（1-40Hz 带通 + 50Hz 陷波，
0.5s-4.5s 共 4s 窗口），保证两层项目的数据口径一致。
"""
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).parent
BCI_ROOT = PROJECT_ROOT.parent


def _find_data_dir():
    """数据目录查找：优先 AIC/data，其次复用 EEG Explorer/data（只读，不修改）。"""
    candidates = [
        PROJECT_ROOT / "data",
        BCI_ROOT / "EEG Explorer" / "data",
    ]
    for d in candidates:
        if d.exists() and any(d.glob("*T.gdf")):
            return d
    return candidates[0]


DATA_DIR = _find_data_dir()
CACHE_DIR = PROJECT_ROOT / "cache"        # epochs 缓存
RESULTS_DIR = PROJECT_ROOT / "results"    # 实验结果（csv/md/图）
CKPT_DIR = PROJECT_ROOT / "checkpoints"  # 最佳模型权重
for _d in (CACHE_DIR, RESULTS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SUBJECT_FILE_TEMPLATE = "A{:02d}{}.gdf"   # {} 为 session 后缀：T=训练场次，E=评测场次
DATA_SESSIONS = ("T", "E")                # 正式协议使用两场次（每被试 216×2=432 trials）
# E 场次 GDF 不含类别事件（仅 783 试次开始标记），真值标签由竞赛官方 A0XE.mat 提供
LABEL_DIR = DATA_DIR / "true_labels_2a"
N_SUBJECTS = 9

# ---------------------------------------------------------------- 信号参数
SFREQ = 250
TMIN = 0.5            # 提示音后 0.5s 开始（避开视觉诱发电位）
TMAX = 4.5            # 到 4.5s，共 4s
WIN_SAMPLES = int(round((TMAX - TMIN) * SFREQ))   # 1000，自动推导避免手写不一致
N_CHANNELS = 22       # BCI IV 2a：22 EEG（另有 3 EOG，按名称剔除）
N_CLASSES = 3         # 左手 / 右手 / 双脚（舌头 772 类别不做）
CLASS_NAMES = ["left_hand", "right_hand", "both_feet"]

LOWCUT = 1.0
HIGHCUT = 40.0
FILTER_ORDER = 4
NOTCH_FREQ = 50.0
NOTCH_Q = 30

# GDF 事件码白名单 -> 训练标签（其余事件：276/277/768/1023/1072/32766/772 全部跳过）
EVENT_TO_LABEL = {"769": 0, "770": 1, "771": 2}
SKIP_EOG_PREFIX = "EOG"
EEG_PREFIX = "EEG-"

# 频带 (Hz)：mu 节律(8-12) 是运动想象 ERD/ERS 的核心频带
FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "mu": (8, 12),
    "beta": (13, 30),
}

# ---------------------------------------------------------------- 训练超参
SEED = 42
SEEDS = [42, 123, 2024, 7, 2025]   # 多种子重复用（LOSO/消融方差大，结论需带误差棒）
USE_EA = True                    # 欧氏对齐开关（无标签跨被试对齐，EA）
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20
VAL_RATIO = 0.2       # 从训练集中再抽 20% 做验证
TEST_RATIO = 0.2      # subject-dependent 划分的测试集比例

# Cropped Training（滑窗裁剪训练）：Schirrmeister 2017 在小样本 EEG 上的核心技巧。
# 每个 4s trial (T=1000) 用滑窗切成多个 2s crop (crop_size=500, stride=64 -> 9 crops，
# 含末端对齐窗)，训练样本量 x9 起到数据增强作用；测试时同一 trial 的所有 crop 预测求平均（软投票）。
# 实测（seed=42 A/B）：本数据管道下裁剪无增益（T 场 eegnet 55.3→55.3、stfa 60.9→59.9；
# T+E 场 eegnet 62.6→61.6），2s 窗丢失全 trial 上下文，故正式协议关闭。
USE_CROPPED = False
CROP_SIZE = 500       # 2s 窗 @250Hz
CROP_STRIDE = 64      # 滑窗步长，1000 样本下产生 9 个 crop（0..448 step 64 + 末端对齐窗 500）

# ---------------------------------------------------------------- 模型超参
F1 = 8                          # 时间卷积基础通道数
DEPTH = 2                       # 空间深度卷积倍数（特征图数 = F1 * DEPTH）
TEMPORAL_KERNELS = (16, 32, 64)  # 多尺度时间卷积核（小核=短时变化，大核=长时过程）
EEGNET_KERNEL = 64              # EEGNet / 单尺度对照的时间卷积核
F_FREQ = 16                     # 频谱分支输出通道数
FUSION_DIM = 64                 # 自适应融合后的特征维度
ATTN_REDUCTION = 4              # SE 类注意力的压缩比
DROPOUT = 0.5
STFT_NFFT = 128
STFT_HOP = 64
# 跨被试对抗损失权重上限（GRL warmup 到该值）。
# 取 0.2 而非 DANN 原文的 1.0：本任务仅 9 个被试、每被试 216 样本，
# 0.5 时对抗项把总损失推高至 ~1.9 且验证集任务精度提前停滞，
# 说明对抗压力过强、损伤了任务特征；弱对齐(0.2)保留跨被试收益、少牺牲任务信息。
DG_LAMBDA_MAX = 0.2

# ---------------------------------------------------------------- 模型容量档
# standard : 完整容量（subject-dependent 正式模型，约 1.3 万参数）
# light    : 小容量档（约 5 千参数，与 EEGNet 的 2947 同量级）。
#   动机：LOSO 只有 8 个训练被试（~1700 trials），standard 档 1.3 万参数
#   会记住"被试个性"而非"运动意图共性"（LOSO v1/v2：STFA ~48-49% < EEGNet ~54%）。
#   压缩通道/融合维度并提高 dropout，验证"同参数预算下时空频谱结构仍优于 EEGNet"。
MODEL_PROFILES = {
    "standard": {
        "F1": 8,          # 时间卷积分支通道数
        "depth": 2,       # 空间深度卷积倍数
        "F_freq": 16,     # 频谱分支输出通道数
        "fusion_dim": 64, # 自适应融合特征维度
        "head_hidden": 32,  # 任务分类头隐层
        "dg_hidden": 64,    # 被试鉴别头隐层
        "dropout": 0.5,
    },
    "light": {
        "F1": 4,
        "depth": 2,
        "F_freq": 8,
        "fusion_dim": 24,
        "head_hidden": 16,
        "dg_hidden": 16,
        "dropout": 0.6,
    },
}
DEFAULT_PROFILE = "standard"


def get_profile(name=DEFAULT_PROFILE):
    """容量档 -> 超参 dict。白名单校验：未知档位直接报错，避免静默走错配置。"""
    if name not in MODEL_PROFILES:
        raise ValueError(f"未知模型容量档: {name}，可选: {list(MODEL_PROFILES)}")
    return dict(MODEL_PROFILES[name])
