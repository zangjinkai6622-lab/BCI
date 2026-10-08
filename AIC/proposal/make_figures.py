# -*- coding: utf-8 -*-
"""申报书插图生成：图1系统架构 / 图2网络结构 / 图3门控机制 /
图4跨被试验证 / 图5实验结果。纯 matplotlib 绘制，中文字体 Microsoft YaHei。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parent.parent / "results"

# 学术风配色：白底 + 灰线 + 单一深蓝强调
C_INPUT = "#FFFFFF"; C_PRE = "#FFFFFF"; C_ALG = "#FFFFFF"
C_NET = "#E9F0F7"; C_OUT = "#FFFFFF"; C_ARM = "#FFFFFF"
C_MAIN = "#1F4E79"; C_ACC = "#1F4E79"; C_FREQ = "#1F4E79"; C_WARN = "#A6322B"
C_BOXLINE = "#595959"; C_MUTE = "#C9D2DC"


def box(ax, x, y, w, h, text, fc, fs=10.5, ec=C_BOXLINE, lw=1.2, bold=False, tc="#1a1a1a"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color="#333333", lw=1.6, ls="-", ms=14, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


# ================================================================ 图1 系统架构
def fig1():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 4.8); ax.axis("off")

    y0, h = 1.9, 1.35
    c1 = box(ax, 0.15, y0, 1.65, h, "EEG 电极帽\n22 导联\n采样 250Hz", C_INPUT, 10.5, bold=True)
    c2 = box(ax, 2.15, y0, 1.95, h, "信号预处理\n1–40Hz 带通\n50Hz 陷波 · 4s 分段", C_PRE, 10)
    c3 = box(ax, 4.45, y0, 1.6, h, "欧氏对齐 EA\n无标签\n跨被试对齐", C_ALG, 10)
    c4 = box(ax, 6.35, y0 - 0.1, 2.45, h + 0.2,
             "STFA-Net 时空频解码\n多尺度时间卷积 · 空间深度卷积\nSTFT 频域分支 · 自适应门控",
             C_NET, 10, ec=C_MAIN, lw=1.8, bold=True)
    c5 = box(ax, 9.15, y0, 1.5, h, "三类意图\n左手 / 右手\n双脚", C_OUT, 10, bold=True)
    c6 = box(ax, 11.0, y0, 1.65, h, "三自由度机械臂\n左移 / 右移\n抓取（开合）", C_ARM, 10, bold=True)
    for a, b in zip([c1, c2, c3, c4, c5], [c2, c3, c4, c5, c6]):
        arrow(ax, (a[0] + 0.83, y0 + h / 2), (b[0] - 0.83, y0 + h / 2))

    # 核心贡献虚线框
    ax.add_patch(Rectangle((4.32, 1.45), 4.62, 2.25, fill=False, ec=C_MAIN,
                           lw=1.6, ls="--", zorder=4))
    ax.text(6.63, 3.95, "本作品核心贡献（算法层，无需采集端/执行端改动）",
            ha="center", fontsize=10.5, color=C_MAIN, fontweight="bold")

    # 底部：跨被试免校准说明
    box(ax, 2.55, 0.3, 8.2, 0.78,
        "新用户免校准：LOSO 验证“8 人训练 → 直接预测第 9 人”，无需逐人采集校准数据",
        C_NET, 10, ec=C_MAIN)
    ax.text(10.83, 3.45, "指令映射", ha="center", fontsize=9, color="#555555")
    ax.set_title("面向神经工程科研/康复实训的跨被试运动想象脑电解码系统总体架构", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_system_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图2 STFA 网络
def fig2():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 7.4); ax.axis("off")

    box(ax, 0.15, 3.15, 1.5, 1.0, "输入 X\n1×22×1000", C_INPUT, 10, bold=True)
    g_c = box(ax, 1.95, 3.15, 1.55, 1.0, "通道注意力门\nx·(1+αc·w)", "#FFFFFF", 9.5)

    # 时空主干（上分支，实线框）
    ax.add_patch(FancyBboxPatch((3.85, 4.35), 4.6, 2.35,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc="white", ec=C_MAIN, lw=1.4, zorder=1))
    ax.text(6.15, 6.42, "时-空主干", ha="center", fontsize=10.5,
            color=C_MAIN, fontweight="bold")
    box(ax, 4.05, 5.35, 2.0, 0.85, "多尺度时间卷积\nk = 16 / 32 / 64", C_NET, 9.5)
    box(ax, 6.25, 5.35, 2.0, 0.85, "拼接 + BN/ELU\nDropout 0.5", "#FFFFFF", 9.5)
    box(ax, 4.05, 4.5, 2.0, 0.75, "空间深度卷积\nDepthwise 22 通道", C_NET, 9.5, bold=True)
    box(ax, 6.25, 4.5, 2.0, 0.75, "BN/ELU + 平均池化", "#FFFFFF", 9.5)
    g_t = box(ax, 4.05, 3.7, 4.2, 0.55, "时间注意力门（αt）：x · (1 + αt·σ(w))", "#FFFFFF", 9.5)

    # 频域分支（下分支，虚线框区分）
    ax.add_patch(FancyBboxPatch((3.85, 0.55), 4.6, 2.35,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc="white", ec=C_MAIN, lw=1.4, linestyle="--", zorder=1))
    ax.text(6.15, 2.62, "频域分支", ha="center", fontsize=10.5,
            color=C_MAIN, fontweight="bold")
    box(ax, 4.05, 1.85, 2.0, 0.7, "STFT 时频谱\nnfft=128 hop=64", C_NET, 9.5)
    box(ax, 6.25, 1.85, 2.0, 0.7, "2D 卷积\nμ/β 节律特征", C_NET, 9.5)
    g_f = box(ax, 4.05, 0.75, 4.2, 0.55, "频域自适应门（αf）：零初始化，训练后可导出", "#FFFFFF", 9.5)

    arrow(ax, (0.9, 3.65), (1.95, 3.65))
    arrow(ax, (3.5, 3.65), (4.05, 5.72))
    arrow(ax, (3.5, 3.65), (4.05, 2.2))
    arrow(ax, (6.05, 5.77), (6.25, 5.77))
    arrow(ax, (7.25, 5.35), (7.25, 5.25))
    arrow(ax, (6.05, 4.88), (6.25, 4.88))
    arrow(ax, (6.15, 4.5), (6.15, 4.28))
    arrow(ax, (6.05, 2.2), (6.25, 2.2))
    arrow(ax, (6.15, 1.85), (6.15, 1.32))

    fus = box(ax, 8.8, 2.75, 1.85, 1.55,
              "自适应融合\nAdaptiveFusion\nsoftmax 可学习权重", C_NET, 9.5, ec=C_MAIN, bold=True)
    arrow(ax, (8.25, 3.97), (8.8, 3.85))
    arrow(ax, (8.25, 1.02), (8.8, 3.2))
    g_feat = box(ax, 10.9, 3.1, 1.7, 0.85, "特征注意力门\nαfeat", "#FFFFFF", 9.5)
    arrow(ax, (10.65, 3.52), (10.9, 3.52))
    head = box(ax, 10.9, 1.75, 1.7, 1.0, "分类头\nLinear→ELU→BN\n→Linear(3)", C_OUT, 9.5)
    arrow(ax, (11.75, 3.1), (11.75, 2.78))
    out = box(ax, 9.3, 0.6, 3.3, 0.8,
              "logits：左手 / 右手 / 双脚\n（交叉熵损失）", C_ARM, 10, bold=True)
    arrow(ax, (11.75, 1.75), (11.75, 1.42))

    ax.annotate("α=0 起步：恒等映射", xy=(2.72, 3.15), xytext=(2.72, 2.7),
                ha="center", fontsize=8.5, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=1.0))
    ax.set_title("STFA-Net 网络结构：时-空-频三分支 + 零初始化自适应门控", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stfa_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图3 门控机制
def fig3():
    gates = pd.read_csv(RES / "gate_values.csv")
    g = gates[gates["model"] == "stfa"].groupby("gate")["alpha"].agg(["mean", "std"]).reindex(
        ["channel_attn", "temporal_attn", "feature_attn", "frequency"])
    labels = ["通道注意力\nαc", "时间注意力\nαt", "特征注意力\nαfeat", "频域分支\nαf"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    # 左：门控结构
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.2, 2.6, 1.0, 0.8, "x", C_INPUT, 12, bold=True)
    # 直通支路
    ax.plot([1.2, 8.6], [3.0, 3.0], color="#333333", lw=2, zorder=1)
    # 注意力支路
    box(ax, 2.0, 4.4, 2.0, 0.8, "压缩-激励\nMLP(·)", "#FFFFFF", 10)
    box(ax, 4.6, 4.4, 1.6, 0.8, "sigmoid\nw∈(0,1)", "#FFFFFF", 10)
    box(ax, 6.8, 4.4, 1.4, 0.8, "× α\n(可学习)", "#FFFFFF", 10, ec=C_MAIN, bold=True)
    ax.text(7.5, 5.55, "α 零初始化：训练第 0 步门控严格为恒等映射",
            ha="center", fontsize=9.5, color=C_MAIN, fontweight="bold")
    arrow(ax, (0.7, 3.4), (0.7, 4.0), color="#666666")
    arrow(ax, (1.2, 4.8), (2.0, 4.8))
    arrow(ax, (4.0, 4.8), (4.6, 4.8))
    arrow(ax, (6.2, 4.8), (6.8, 4.8))
    arrow(ax, (7.5, 4.4), (7.5, 3.55), color="#333333")
    ax.add_patch(plt.Circle((7.5, 3.0), 0.16, fc="white", ec="#333333", lw=1.6, zorder=4))
    ax.text(7.5, 3.0, "+", ha="center", va="center", fontsize=13, zorder=5)
    ax.add_patch(plt.Circle((9.0, 3.0), 0.16, fc="white", ec="#333333", lw=1.6, zorder=4))
    ax.text(9.0, 3.0, "×", ha="center", va="center", fontsize=13, zorder=5)
    arrow(ax, (7.66, 3.0), (8.84, 3.0))
    box(ax, 9.35, 2.6, 0.55, 0.8, "x′", C_OUT, 12, bold=True)
    ax.text(5.0, 2.05, r"$x' = x \odot \left(1 + \alpha \cdot \sigma(Wx)\right)$",
            fontsize=14, ha="center", color="#1a1a1a")
    ax.text(5.0, 1.25, "α>0 增强该模块 · α≈0 关闭该模块 · α<0 抑制其噪声",
            fontsize=10, ha="center", color="#555555")
    ax.text(5.0, 0.55, "强度由数据梯度自主裁决，无需人工设定开关",
            fontsize=10, ha="center", color="#555555")

    # 右：学到的 α（单一深蓝，正负仅由方向表示）
    ax2 = axes[1]
    colors = [C_MAIN] * 4
    ax2.bar(np.arange(4), g["mean"], yerr=g["std"], capsize=5,
            color=colors, alpha=0.85, edgecolor=C_MAIN)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("训练后门控强度 α（5 seeds 均值±标准差）")
    lo = min(-0.2, float(g["mean"].min() - g["std"].max()) - 0.15)
    hi = max(0.35, float(g["mean"].max() + g["std"].max()) + 0.2)
    ax2.set_ylim(lo, hi)
    ax2.set_title("模型用梯度对各模块“投票”：α 的正负与大小代表增强/抑制强度", fontsize=10.5)
    for i, (m, s) in enumerate(zip(g["mean"], g["std"])):
        ax2.text(i, m - 0.07 if m < 0 else m + 0.03, f"{m:.2f}",
                 ha="center", va="top" if m < 0 else "bottom", fontsize=10)
    fig.suptitle("模块可靠性自适应门控（MAG）机制与训练后 α 实测值", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gating_mechanism.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图4 跨被试
def fig4():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 5.4); ax.axis("off")

    # 左：LOSO
    xs = [0.7, 2.4, 4.1]; ys = [3.9, 2.2, 0.7]
    pos = [(x, y) for y in ys for x in xs]
    labels = [f"S{i}" for i in range(1, 10)]
    for i, ((x, y), lb) in enumerate(zip(pos, labels)):
        if i == 4:
            box(ax, x, y, 0.95, 0.7, lb, C_NET, 11, ec=C_MAIN, lw=1.8, bold=True)
        else:
            box(ax, x, y, 0.95, 0.7, lb, "#FFFFFF", 11, bold=True)
    ax.text(2.4, 4.9, "LOSO 留一被试验证（9 折）", ha="center",
            fontsize=11.5, fontweight="bold")
    ax.text(2.4, 0.05, "浅蓝=未见测试被试（模型训练中从未出现）\n其余 8 人=训练+验证，逐人轮换共 9 折",
            ha="center", fontsize=9, color="#555555")

    arrow(ax, (5.4, 2.55), (6.1, 2.55), lw=2)

    # 右：EA + GRL（任务支路深蓝，对抗支路灰色）
    box(ax, 6.2, 4.2, 2.3, 0.85, "欧氏对齐（无标签）\nX←R_s^(-1/2)·X", "#FFFFFF", 9.5)
    box(ax, 6.55, 2.2, 1.6, 0.95, "特征提取器\nSTFA 编码器", C_NET, 10, ec=C_MAIN, bold=True)
    arrow(ax, (7.35, 4.2), (7.35, 3.18))
    fpt = (8.15, 2.68)
    # 任务分类器
    box(ax, 9.3, 3.35, 2.0, 0.85, "任务分类器\n左手/右手/双脚", "#FFFFFF", 10)
    arrow(ax, fpt, (9.3, 3.78), color=C_MAIN, lw=2)
    ax.text(9.9, 3.05, "L_task", fontsize=10, color=C_MAIN, fontweight="bold")
    # GRL + 鉴别器
    box(ax, 9.3, 1.25, 1.55, 0.8, "GRL\n梯度反转", "#FFFFFF", 10, ec="#808080")
    box(ax, 11.0, 1.25, 1.65, 0.8, "被试鉴别器\n8 类（留一后）", "#FFFFFF", 10)
    arrow(ax, fpt, (9.3, 1.65), color="#808080", lw=2)
    arrow(ax, (10.85, 1.65), (11.0, 1.65), color="#808080", lw=2)
    ax.text(10.3, 0.85, "L_domain（λ warmup 0→0.2）", fontsize=9.5,
            color="#808080", ha="center")
    ax.text(8.9, 0.35, "对抗目标：特征中“能识别被试身份”的信息被抹除，只保留运动意图共性",
            fontsize=9.5, ha="center", color="#555555")
    ax.set_title("跨被试免校准泛化：LOSO 协议 + 欧氏对齐(EA) + 梯度反转域对抗(GRL)", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_cross_subject.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图5 实验结果
def fig5():
    comp = pd.read_csv(RES / "comparison.csv")
    abl = pd.read_csv(RES / "ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # 左：对比
    names = ["RF", "SVM", "LR", "EEGNet", "STFA-Net", "STFA-DGNet"]
    tab = comp.set_index("model")
    accs, stds, cols = [], [], []
    for n in ["RF", "SVM", "LR", "eegnet", "stfa", "stfa_dg"]:
        accs.append(tab.loc[n, "accuracy_mean"] * 100)
        stds.append(tab.loc[n, "accuracy_std"] * 100)
        cols.append(C_MAIN if n == "stfa" else ("#8FAADC" if n == "stfa_dg" else C_MUTE))
    bars = axes[0].bar(names, accs, yerr=stds, capsize=4, color=cols,
                       edgecolor="#7A7A7A", linewidth=0.8, alpha=0.95)
    axes[0].axhline(100 / 3, color="#808080", ls="--", lw=1.2)
    axes[0].text(5.4, 34.5, "随机水平 33.3%", color="#595959", fontsize=9, ha="right")
    for b, a in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width() / 2, a + 1.2, f"{a:.1f}",
                     ha="center", fontsize=9.5)
    axes[0].set_ylabel("测试集准确率 (%)")
    axes[0].set_ylim(30, max(70, max(accs) + 6))
    axes[0].set_title("被试内对比（5 seeds 均值±标准差）", fontsize=11)
    axes[0].tick_params(axis="x", rotation=20)

    # 右：消融
    tags = ["stfa_full", "wo_multiscale_time", "wo_spatial",
            "wo_frequency", "wo_attention"]
    labs = ["完整模型", "去多尺度\n时间", "去空间\n卷积", "去频域\n分支", "去注意力"]
    sub = abl.set_index("ablation").loc[tags]
    acc = sub["accuracy_mean"].values * 100
    sdv = sub["accuracy_std"].values * 100
    full = acc[0]
    cols2 = [C_MAIN] + [C_MUTE] * 4
    bars = axes[1].bar(labs, acc, yerr=sdv, capsize=4, color=cols2,
                       edgecolor="#7A7A7A", linewidth=0.8, alpha=0.95)
    axes[1].axhline(full, color="#333333", ls="--", lw=1)
    axes[1].text(-0.45, full + 0.4, f"完整 {full:.1f}%", ha="left", fontsize=9)
    for b, v in zip(bars, acc):
        delta = v - full
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1.2,
                     f"{v:.1f}\n({delta:+.1f})", ha="center", fontsize=8.8)
    axes[1].set_ylabel("测试集准确率 (%)")
    axes[1].set_ylim(min(35, min(acc) - 6), max(70, max(acc) + 5))
    drop = full - acc[2]
    axes[1].set_title(f"消融实验：空间卷积为硬依赖（−{drop:.1f}%），门控模块由数据自适应", fontsize=11)
    fig.suptitle("STFA-Net 定量实验结果", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 图0 图形摘要
def _icon_head_cap(ax, cx, cy, s=1.0):
    """佩戴电极帽的用户头部简笔画。s 为缩放系数。"""
    skin = "#F6DCC8"; cap_col = "#1F4E79"; elec = "#E8A33D"
    # 肩膀
    ax.add_patch(FancyBboxPatch((cx - 0.62*s, cy - 1.02*s), 1.24*s, 0.34*s,
                 boxstyle="round,pad=0.01,rounding_size=0.12", fc="#DCE6F1",
                 ec="#8CA3BD", lw=1.0, zorder=2))
    # 头部
    head = plt.Circle((cx, cy - 0.28*s), 0.52*s, fc=skin, ec="#8A6A50",
                      lw=1.2, zorder=3)
    ax.add_patch(head)
    # 电极帽：头顶圆弧
    arc = matplotlib.patches.Arc((cx, cy - 0.28*s), 1.06*s, 1.06*s, angle=0,
                                 theta1=22, theta2=158, lw=8*s, color=cap_col,
                                 zorder=4, capstyle="round")
    ax.add_patch(arc)
    # 电极点（沿弧线）
    for deg in (40, 62, 90, 118, 140):
        rad = np.radians(deg)
        ex = cx + 0.53*s*np.cos(rad)
        ey = cy - 0.28*s + 0.53*s*np.sin(rad)
        ax.add_patch(plt.Circle((ex, ey), 0.055*s, fc=elec, ec="white",
                                lw=0.6, zorder=5))


def _icon_arm(ax, cx, cy, s=1.0):
    """机械臂简笔画：底座 + 两段臂 + 夹持器。"""
    steel = "#5B7FA6"; joint = "#2E5984"
    # 底座
    ax.add_patch(Rectangle((cx - 0.34*s, cy - 0.92*s), 0.68*s, 0.2*s,
                 fc="#44505C", ec="#2F3840", lw=0.9, zorder=2))
    p0 = (cx, cy - 0.72*s)
    p1 = (cx - 0.30*s, cy - 0.12*s)
    p2 = (cx + 0.34*s, cy + 0.42*s)
    for a, b in ((p0, p1), (p1, p2)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=steel, lw=7*s,
                solid_capstyle="round", zorder=3, solid_joinstyle="round")
    for p in (p0, p1, p2):
        ax.add_patch(plt.Circle(p, 0.115*s, fc=joint, ec="white", lw=1.0, zorder=4))
    # 夹持器（两爪）
    tip_l = (p2[0] + 0.22*s, p2[1] + 0.22*s)
    tip_r = (p2[0] + 0.22*s, p2[1] - 0.22*s)
    ax.plot([p2[0], tip_l[0]], [p2[1], tip_l[1]], color="#C9A227", lw=4.5*s,
            solid_capstyle="round", zorder=3)
    ax.plot([p2[0], tip_r[0]], [p2[1], tip_r[1]], color="#C9A227", lw=4.5*s,
            solid_capstyle="round", zorder=3)


def _icon_house(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx-0.26*s, cy-0.30*s), 0.52*s, 0.34*s, fc="#DCE6F1",
                 ec="#5B7FA6", lw=1.1, zorder=3))
    roof = plt.Polygon([(cx-0.34*s, cy+0.04*s), (cx, cy+0.34*s),
                        (cx+0.34*s, cy+0.04*s)], fc="#1F4E79", ec="none", zorder=3)
    ax.add_patch(roof)
    ax.add_patch(Rectangle((cx-0.07*s, cy-0.30*s), 0.14*s, 0.17*s, fc="white",
                 ec="#5B7FA6", lw=0.9, zorder=4))


def _icon_cross(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx-0.30*s, cy-0.26*s), 0.60*s, 0.52*s, fc="white",
                 ec="#5B7FA6", lw=1.1, zorder=2))
    ax.add_patch(Rectangle((cx-0.06*s, cy-0.18*s), 0.12*s, 0.36*s, fc="#A6322B",
                 ec="none", zorder=3))
    ax.add_patch(Rectangle((cx-0.18*s, cy-0.06*s), 0.36*s, 0.12*s, fc="#A6322B",
                 ec="none", zorder=3))


def _icon_book(ax, cx, cy, s=1.0):
    left = plt.Polygon([(cx-0.34*s, cy-0.22*s), (cx, cy-0.10*s),
                        (cx, cy+0.24*s), (cx-0.34*s, cy+0.12*s)],
                       closed=True, fc="#DCE6F1", ec="#5B7FA6", lw=1.0, zorder=3)
    right = plt.Polygon([(cx+0.34*s, cy-0.22*s), (cx, cy-0.10*s),
                         (cx, cy+0.24*s), (cx+0.34*s, cy+0.12*s)],
                        closed=True, fc="#C6D7EA", ec="#5B7FA6", lw=1.0, zorder=3)
    ax.add_patch(left); ax.add_patch(right)
    ax.plot([cx, cx], [cy-0.10*s, cy+0.24*s], color="#5B7FA6", lw=1.2, zorder=4)


def fig0():
    """系统总体流程图：朴素技术风格，黑白为主、单一蓝色强调。"""
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 5.5); ax.axis("off")
    ax.set_facecolor("white")

    # 颜色
    c_box = "#333333"       # 普通方框边框
    c_text = "#1a1a1a"      # 正文文字
    c_main = "#1F4E79"     # 强调色（仅用于核心模型 + 标题）
    c_arr = "#555555"      # 箭头
    c_fb = "#999999"       # 反馈线

    # ---------- 标题
    ax.text(5.5, 5.25, "运动想象脑电解码系统总体流程",
            ha="center", fontsize=14, fontweight="bold", color=c_main)

    # ---------- 辅助：绘制简单方框
    def box(x, y, w, h, title, sub="", highlight=False):
        fc = "#E8F0F7" if highlight else "white"
        ec = c_main if highlight else c_box
        lw = 1.8 if highlight else 1.2
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.01,rounding_size=0.06",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ty = y + h - 0.28 if sub else y + h/2
        ax.text(x+w/2, ty, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=c_main if highlight else c_text,
                zorder=3)
        if sub:
            ax.text(x+w/2, y+0.27, sub, ha="center", va="center", fontsize=8.5,
                    color="#555555", zorder=3, linespacing=1.3)

    def arrow_h(x1, x2, y, color=c_arr, lw=1.5, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                     mutation_scale=12, color=color, lw=lw, linestyle=ls,
                     zorder=1))

    # ---------- 第一行：数据采集 → 预处理 → 模型推理 → 输出
    y1 = 3.35; h1 = 1.15
    box(0.3, y1, 2.0, h1, "脑电信号采集", "22 导联电极帽\n250 Hz 采样")
    box(2.85, y1, 2.0, h1, "信号预处理", "1-40 Hz 带通\n分段截取 2 s")
    box(5.4, y1, 2.6, h1, "STFA-Net 解码", "时空频特征提取\nEA 无标签对齐", highlight=True)
    box(8.55, y1, 2.15, h1, "意图分类输出", "左手 / 右手 / 双脚\n后验概率")

    arrow_h(2.32, 2.83, y1+h1/2)
    arrow_h(4.87, 5.38, y1+h1/2)
    arrow_h(8.02, 8.53, y1+h1/2)

    # ---------- 第二行：机械臂执行 + 反馈
    y2 = 1.45; h2 = 1.05
    box(3.5, y2, 4.0, h2, "机械臂执行动作", "左移 / 右移 / 抓取")
    arrow_h(9.62, 9.62, y1)  # 竖直转角用
    # 从意图输出向下到机械臂
    ax.add_patch(FancyArrowPatch((9.62, y1), (9.62, y2+h2), arrowstyle="-|>",
                 mutation_scale=12, color=c_arr, lw=1.5, zorder=1))
    # 横向到机械臂框
    arrow_h(9.62, 7.52, y2+h2/2)

    # ---------- 反馈回路
    fb = FancyArrowPatch((3.48, y2+h2/2), (1.3, y1), arrowstyle="-|>",
                         mutation_scale=11, color=c_fb, lw=1.2, ls=(0,(4,3)),
                         connectionstyle="arc3,rad=0.25", zorder=0)
    ax.add_patch(fb)
    ax.text(1.8, 2.35, "视觉反馈", ha="center", fontsize=8.5, color=c_fb,
            style="italic", rotation=0)

    # ---------- 底部：应用场景（纯文字列表，不用色块）
    ax.text(5.5, 0.70, "应用场景：残障人士辅助操控  ·  康复训练意图触发  ·  神经工程教学实验",
            ha="center", fontsize=9.5, color="#444444")
    ax.text(5.5, 0.30, "部署特点：免校准（EA 秒级对齐）  ·  轻量模型（1.3 万参数）  ·  CPU 实时推理（约 3 ms）",
            ha="center", fontsize=9.5, color="#444444")

    fig.savefig(OUT / "fig0_graphical_abstract.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


for f in (fig0, fig1, fig2, fig3, fig4, fig5):
    f(); print("done:", f.__name__)
print("all figures ->", OUT)
