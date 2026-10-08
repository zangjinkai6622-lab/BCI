# -*- coding: utf-8 -*-
"""演示视频渲染：真实 LOSO 模型 + 真实 EEG 流式推理回放 + 2D 机械臂动画。

输出：demo/演示视频_机械臂意念控制的自适应脑电解码算法.mp4（1280x720）
素材：demo_data.npz（被试 1 E 场真实试验 + 逐帧概率）、audio/*.wav（中文旁白）、
      proposal/figures/*.png（架构/结果图）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow, Circle
from PIL import Image
from moviepy import (ImageClip, VideoClip, AudioFileClip, CompositeAudioClip,
                     concatenate_videoclips)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "proposal" / "figures"
AUD = HERE / "audio"
FPS = 20
W, H = 1280, 720

CLASS_NAME = ["左手", "右手", "双脚"]
ACTION_NAME = ["左移", "右移", "抓取"]
CLASS_COLOR = ["#4C72B0", "#DD8452", "#55A467"]
BG = "#0E1B2A"; PANEL = "#16263C"; GRID = "#2E4363"; TXT = "#EAF0FA"; MUTED = "#93A7C4"

z = np.load(HERE / "demo_data.npz", allow_pickle=True)
trials, labels, probs = z["trials"], z["labels"], z["probs"]   # (9,22,1000),(9,),(9,40,3)
N_TRIALS = len(trials)

# 场景时长（秒）= 旁白时长 + 尾部停顿（edge-tts Yunxi 配音实测，总时长 ~184s）
DUR = {"s0": 13.5, "s1": 30.5, "s2": 16.5, "s3": 64.0, "s4": 22.0, "s5": 18.0, "s6": 19.5}
SUB = {
    "s1": "运动想象脑机接口帮助 ALS、高位截瘫、脑卒中患者用意念控制设备。三大瓶颈：①标注样本稀缺，每用户仅百余试验；②新用户校准需 20–30 分钟；③跨被试差异大，换人即可能失效。",
    "s2": "四层架构：信号采集 → 预处理（1–40Hz 带通、50Hz 陷波）→ STFA-Net 解码（零初始化模块可靠性门控 MAG）→ 机械臂控制；无标签欧氏对齐用于批处理适配。",
    "s4": "9 被试 × 2 场次 × 3888 试验 × 5 seeds：STFA-Net 66.02%，较 EEGNet +2.76pp；消融证实空间卷积为硬依赖（−17.71pp）。",
    "s5": "传统流程：120 个标签试验、约 19 分钟采集 + 个性化训练；本方案：用 216 个无标签试验做 EA 批处理适配；不需要人工标签。",
    "s6": "核心创新：零初始化门控让数据裁决模块、无标签对齐减少用户标注。后续：无监督域适应 + 在线自适应 + 真实康复试点。",
}

# ------------------------------------------------------------ 基础绘制
fig_cache = {}

def _base_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1280); ax.set_ylim(0, 720)
    ax.axis("off"); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    return fig, ax

def _panel(ax, x, y, w, h, fc=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=2,rounding_size=14",
                                fc=fc, ec=GRID, lw=1.2))

def _wrap(s, n):
    out, line = [], ""
    for ch in s:
        line += ch
        if len(line) >= n:
            out.append(line); line = ""
    if line: out.append(line)
    return out

def _subtitle(ax, text):
    _panel(ax, 60, 24, 1160, 86, fc="#102033")
    for i, line in enumerate(_wrap(text, 42)[:2]):
        ax.text(640, 88 - i * 30, line, ha="center", va="center", fontsize=14.5,
                color=TXT, family="Microsoft YaHei", linespacing=1.5)

def _scene_header(ax, title, tag):
    ax.add_patch(Rectangle((0, 648), 1280, 72, fc="#13293F"))
    ax.add_patch(Rectangle((0, 648), 10, 72, fc="#4C9BE8"))
    ax.text(42, 684, title, ha="left", va="center", fontsize=24, color="white",
            family="Microsoft YaHei", fontweight="bold")
    ax.text(1238, 684, tag, ha="right", va="center", fontsize=13, color=MUTED,
            family="Microsoft YaHei")

def _fig_frame(key, draw):
    """静态场景只渲染一次并缓存为 HxWx3 数组。"""
    if key not in fig_cache:
        fig, ax = _base_fig()
        draw(fig, ax)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        fig_cache[key] = buf
        plt.close(fig)
    return fig_cache[key]

def _img_ax(fig, rect, path):
    ax = fig.add_axes(rect); ax.axis("off")
    img = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(img); return ax

# ------------------------------------------------------------ 各静态场景
def draw_s0(fig, ax):
    for i, c in enumerate(["#0B1626", "#0E1B2A", "#10243C"]):
        ax.add_patch(Rectangle((0, i * 240), 1280, 240, fc=c, ec="none"))
    ax.add_patch(Rectangle((340, 250), 600, 3, fc="#4C9BE8"))
    ax.text(640, 360, "机械臂意念控制的", ha="center", va="center",
            fontsize=44, color="white", fontweight="bold")
    ax.text(640, 286, "自适应脑电解码算法", ha="center", va="center",
            fontsize=44, color="#7FC4FF", fontweight="bold")
    ax.text(640, 196, "运动想象脑电解码 · 跨被试免校准 · 三自由度辅助机械臂",
            ha="center", fontsize=18, color=MUTED)
    ax.text(640, 130, "全球校园人工智能算法精英大赛 · 算法创新赛（AI＋学科交叉）",
            ha="center", fontsize=15, color="#9FC3E8")
    ax.text(640, 80, "公开基准 BCI Competition IV-2a ｜ 9 被试 × 2 场次 × 3888 试验",
            ha="center", fontsize=12.5, color=MUTED)

def draw_s1(fig, ax):
    _scene_header(ax, "一、背景与痛点", "神经工程 / 康复工程")
    cards = [
        ("样本稀缺", "每名用户仅约 200 次\n4s 有标签试验", "深度网络易过拟合"),
        ("校准耗时", "新用户需 20–30 分钟\n有标签标定", "占用康复黄金时间"),
        ("跨被试漂移", "脑形态/阻抗差异大\n换人即可能失效", "实验实训难标准化"),
    ]
    for i, (t1, t2, t3) in enumerate(cards):
        x = 70 + i * 400
        _panel(ax, x, 330, 340, 250)
        ax.add_patch(Circle((x + 170, 518), 42, fc=["#4C72B0", "#DD8452", "#C44E52"][i], alpha=0.9))
        ax.text(x + 170, 518, str(i + 1), ha="center", va="center", fontsize=28,
                color="white", fontweight="bold")
        ax.text(x + 170, 432, t1, ha="center", fontsize=22, color="white", fontweight="bold")
        for j, ln in enumerate(t2.split("\n")):
            ax.text(x + 170, 392 - j * 26, ln, ha="center", fontsize=15, color=TXT)
        ax.text(x + 170, 332 - 14, t3, ha="center", fontsize=12.5, color=MUTED)
    ax.text(640, 268, "服务对象：ALS、高位截瘫、脑卒中患者的运动功能重建",
            ha="center", fontsize=17, color="#7FC4FF")
    ax.text(640, 220, "动作映射：左手 → 左移    右手 → 右移    双脚 → 抓取",
            ha="center", fontsize=16, color=TXT)
    _subtitle(ax, SUB["s1"])

def draw_s2(fig, ax):
    _scene_header(ax, "二、系统总体架构", "四层架构")
    _img_ax(fig, [0.04, 0.16, 0.92, 0.72], FIG / "fig1_system_architecture.png")
    _subtitle(ax, SUB["s2"])

def draw_s4(fig, ax):
    _scene_header(ax, "三、实验结果", "3888 试验 · 5 seeds")
    _img_ax(fig, [0.03, 0.18, 0.58, 0.66], FIG / "fig5_results.png")
    _panel(ax, 810, 150, 400, 440)
    bullets = [
        ("被试内 STFA-Net", "66.02 ± 1.22%"),
        ("EEGNet 基线", "63.26 ± 1.64%"),
        ("相对提升", "+2.76 pp"),
        ("传统 ML（LR/RF/SVM）", "48.97–51.29%"),
        ("去空间卷积", "−17.71 pp（硬依赖）"),
        ("去多尺度时间", "−1.00 pp"),
        ("CPU 推理", "≈3.1 ms / 4s 片段"),
    ]
    for i, (k, v) in enumerate(bullets):
        y = 548 - i * 60
        ax.text(836, y, k, ha="left", va="center", fontsize=14.5, color=TXT)
        ax.text(1184, y, v, ha="right", va="center", fontsize=15.5,
                color="#7FC4FF", fontweight="bold")
    _subtitle(ax, SUB["s4"])

def draw_s5(fig, ax):
    _scene_header(ax, "四、新用户：有标签校准 vs 无标签适配", "实测口径")
    _panel(ax, 90, 200, 520, 380, fc="#2A1E22")
    _panel(ax, 670, 200, 520, 380, fc="#162A1E")
    ax.text(350, 544, "传统有标签校准", ha="center", fontsize=21,
            color="#E89A8A", fontweight="bold")
    ax.text(930, 544, "无标签批处理适配", ha="center", fontsize=21,
            color="#8FD8A8", fontweight="bold")
    left = ["标签试验：120 个（3 类 × 40）", "范式采集：约 18.6 分钟",
            "个性化训练：CPU 约 45 s", "STFA 精度：56.79 ± 13.05%"]
    right = ["人工标签：0 个", "适配数据：216 个无标签试验", "EA 对齐：22.5 ms（216 无标签试验）",
             "LOSO 精度：52.48 ± 15.13%", "IV-2b 预训练：9 被试迁移 +0.91 pp"]
    for i, s in enumerate(left):
        ax.text(120, 470 - i * 78, "•  " + s, ha="left", va="center",
                fontsize=15.5, color="#EBCFC8")
    for i, s in enumerate(right):
        ax.text(700, 470 - i * 62, "•  " + s, ha="left", va="center",
                fontsize=14.5, color="#CFE8D6")
    ax.text(640, 150, "减少人工标签，不等同于零准备实时使用",
            ha="center", fontsize=17, color="#7FC4FF")
    _subtitle(ax, SUB["s5"])

def draw_s6(fig, ax):
    _scene_header(ax, "五、总结与展望", "")
    _panel(ax, 110, 330, 1060, 250)
    inn = [
        "① 零初始化模块可靠性门控 MAG：x·(1+αw)，α 可导出，由数据裁决模块强度",
        "② 无标签欧氏对齐 EA：新用户 22.5 ms 完成适配，零标签、零校准",
        "③ 五种子 + 消融 + LOSO 全量留档，负面结果诚实报告",
    ]
    for i, s in enumerate(inn):
        ax.text(150, 520 - i * 70, s, ha="left", va="center", fontsize=16.5, color=TXT)
    ax.text(640, 258, "展望：无监督域适应 · 个性化在线自适应 · 真实康复场景试点",
            ha="center", fontsize=17, color="#7FC4FF")
    ax.text(640, 130, "感谢观看！", ha="center", fontsize=40, color="white",
            fontweight="bold")
    _subtitle(ax, SUB["s6"])

# ------------------------------------------------------------ 演示场景（核心）
TRIAL_T = 7.11
CUE, IMAG, LOCK = 1.0, 4.0, 0.5

def _draw_arm(ax, pose, grip, cx=962, base_y=118, scale=1.0):
    """二连杆机械臂 + 末端夹爪。pose: (肩角°, 肘角°)；grip 0..1 闭合度。"""
    l1, l2 = 118 * scale, 104 * scale
    th1 = np.deg2rad(pose[0]); th2 = np.deg2rad(pose[1])
    sx, sy = cx, base_y + 34
    ex = sx + l1 * np.cos(th1); ey = sy + l1 * np.sin(th1)
    wx = ex + l2 * np.cos(th1 + th2); wy = ey + l2 * np.sin(th1 + th2)
    # 底座
    s = scale
    ax.add_patch(Rectangle((cx - 60 * s, base_y), 120 * s, 34, fc="#2B4565",
                           ec="#4C9BE8", lw=1.5))
    ax.add_patch(Circle((sx, sy), 13 * s, fc="#4C9BE8"))
    for (p1, p2, w) in [((sx, sy), (ex, ey), 12 * s), ((ex, ey), (wx, wy), 9 * s)]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#8FB8E8", lw=w,
                solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((ex, ey), 9 * s, fc="#5E8BC8", zorder=6))
    # 夹爪
    open_a = 0.5 - 0.34 * grip
    wrist_ang = th1 + th2
    for sgn in (-1, 1):
        ga = wrist_ang + sgn * open_a
        gx = wx + 30 * s * np.cos(ga); gy = wy + 30 * s * np.sin(ga)
        ax.plot([wx, gx], [wy, gy], color="#D8B35A", lw=5 * s,
                solid_capstyle="round", zorder=7)
    return wx, wy

def demo_frame(t):
    fig, ax = _base_fig()
    ax.set_xlim(0, 1280); ax.set_ylim(0, 720)
    _scene_header(ax, "新用户免校准实时推理演示",
                  "LOSO：模型仅用其余 8 名被试训练 · 被试 1 E 场无标签适配后回放")
    # ---- 左：22 导流式 EEG（嵌套坐标，独立 xlim，不裁切主坐标的面板布局）
    _panel(ax, 24, 118, 742, 508)
    ax.text(48, 602, "22 导脑电（μV，去趋势）", fontsize=14, color=TXT, va="center")
    ax.text(742, 602, "缓冲：4 s 滑动窗口", fontsize=11.5, color=MUTED, ha="right", va="center")
    i = min(int(t // TRIAL_T), N_TRIALS - 1)
    tl = t - i * TRIAL_T
    n_show = int(np.clip((tl - CUE) / 0.1, 0, 40)) if tl >= CUE else 0
    pos = n_show * 25
    eax = fig.add_axes([52 / 1280, 168 / 720, 688 / 1280, 418 / 720])
    eax.set_facecolor("none")
    yy = trials[i]
    scale_v = 0.06               # μV → 像素（峰值 ~300μV × 0.06 ≈ 18px，不超出 21.5px 道距）
    for ch in range(22):
        base = 566 - ch * 21.5
        if pos > 1:
            eax.plot(np.arange(pos) / 250.0 + 0.12, yy[ch, :pos] * scale_v + base,
                     color="#79AEE6", lw=0.55, zorder=3)
        eax.axhline(base, color=GRID, lw=0.5, zorder=1)
    eax.set_xlim(0, 4.25); eax.set_ylim(75, 600)
    eax.set_xticks([]); eax.set_yticks([])
    for sp in eax.spines.values(): sp.set_visible(False)
    # 缓冲进度条（主坐标像素定位）
    ax.add_patch(Rectangle((48, 132), 688 * (pos / 1000), 5, fc="#4C9BE8"))
    ax.add_patch(Rectangle((48, 132), 688, 5, fc="none", ec=GRID, lw=0.8))
    ax.text(48, 150, f"已接收 {pos/250:.1f} s / 4.0 s", fontsize=11, color=MUTED, va="center")

    # ---- 右上：线索与状态
    _panel(ax, 790, 470, 466, 156)
    true_c = int(labels[i])
    if tl < CUE:
        cue_txt, st_txt, st_col = "准备 …", "等待线索", MUTED
    elif tl < CUE + IMAG:
        cue_txt, st_txt, st_col = f"请想象：{CLASS_NAME[true_c]}", "信号采集中 …", "#7FC4FF"
    elif tl < CUE + IMAG + LOCK:
        cue_txt, st_txt, st_col = f"请想象：{CLASS_NAME[true_c]}", "指令锁定", "#E8C45A"
    else:
        pred_c = int(probs[i][-1].argmax())
        cue_txt = f"请想象：{CLASS_NAME[true_c]}"
        st_txt = f"执行动作：{ACTION_NAME[pred_c]}"
        st_col = "#8FD8A8"
    ax.text(818, 596, "线索", fontsize=12.5, color=MUTED, va="center")
    ax.text(818, 556, cue_txt, fontsize=21, color=TXT, va="center", fontweight="bold")
    ax.text(818, 506, "状态", fontsize=12.5, color=MUTED, va="center")
    ax.text(1010, 506, st_txt, fontsize=16, color=st_col, va="center", fontweight="bold")
    ax.text(1232, 596, f"试验 {i+1}/{N_TRIALS}", fontsize=12.5, color=MUTED,
            ha="right", va="center")

    # ---- 右中：三类概率条
    _panel(ax, 790, 312, 466, 142)
    ax.text(818, 424, "动作后验概率", fontsize=13.5, color=TXT, va="center")
    pv = probs[i][min(n_show, 39)] if pos > 0 else np.ones(3) / 3
    bar_w_max = 250
    for c in range(3):
        y = 392 - c * 38
        ax.text(818, y, CLASS_NAME[c], fontsize=13.5, color=TXT, va="center")
        ax.add_patch(Rectangle((878, y - 10), bar_w_max, 18, fc="#1D3250"))
        ax.add_patch(Rectangle((878, y - 10), bar_w_max * float(pv[c]), 18,
                               fc=CLASS_COLOR[c]))
        ax.text(1146, y, f"{pv[c]*100:5.1f}%", fontsize=12.5, color=TXT,
                va="center", family="monospace")

    # ---- 右下：机械臂
    _panel(ax, 790, 118, 466, 186)
    rest = (115.0, -70.0)
    targets = {0: (152.0, -42.0), 1: (28.0, 42.0), 2: (100.0, -140.0)}
    pred_c = int(probs[i][-1].argmax())
    act_start = CUE + IMAG + LOCK
    if tl < act_start:
        pose, grip = rest, 0.0
    else:
        u = np.clip((tl - act_start) / 1.4, 0, 1)
        u = 0.5 - 0.5 * np.cos(np.pi * u)       # ease in-out
        tgt = targets[pred_c]
        pose = tuple(rest[k] + (tgt[k] - rest[k]) * u for k in range(2))
        if pred_c == 2:
            grip = np.clip((tl - act_start - 1.0) / 1.0, 0, 1)
        else:
            grip = 0.0
    _draw_arm(ax, pose, grip, scale=0.6)
    ax.text(1032, 136, f"机械臂 → {ACTION_NAME[pred_c] if tl >= act_start else '待机'}",
            fontsize=13, color=("#8FD8A8" if tl >= act_start else MUTED), va="center")

    # ---- 字幕（旁白两段定时）
    sub_a = "模型仅在其余 8 名被试上训练；当前用户先完成无标签对齐，再以 4 秒滑动窗口持续推理。"
    sub_b = "概率随信号进入逐步升高并锁定，机械臂完成左移/右移/抓取，形成想象—解码—执行—反馈闭环。"
    if t < 24.8:
        _subtitle(ax, sub_a)
    elif 25.2 <= t < 43.0:
        _subtitle(ax, sub_b)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf

# ------------------------------------------------------------ 合成
def make_clip(key, draw):
    frame = _fig_frame(key, draw)
    return ImageClip(frame).with_duration(DUR[key])

def main():
    scenes = [
        ("s0", make_clip("s0", draw_s0), "s0_title.wav"),
        ("s1", make_clip("s1", draw_s1), "s1_background.wav"),
        ("s2", make_clip("s2", draw_s2), "s2_architecture.wav"),
    ]
    demo = VideoClip(demo_frame, duration=DUR["s3"])
    a_a = AudioFileClip(str(AUD / "s3_demo_a.wav")).with_start(0.4)
    a_b = AudioFileClip(str(AUD / "s3_demo_b.wav")).with_start(25.5)
    demo = demo.with_audio(CompositeAudioClip([a_a, a_b]))
    scenes.append(("s3", demo, None))
    scenes += [
        ("s4", make_clip("s4", draw_s4), "s4_results.wav"),
        ("s5", make_clip("s5", draw_s5), "s5_calib.wav"),
        ("s6", make_clip("s6", draw_s6), "s6_summary.wav"),
    ]
    clips = []
    for key, clip, wav in scenes:
        if wav:
            clip = clip.with_audio(AudioFileClip(str(AUD / wav)))
        clips.append(clip)
    final = concatenate_videoclips(clips, method="chain")
    out = HERE / "演示视频_机械臂意念控制的自适应脑电解码算法.mp4"
    final.write_videofile(
        str(out), fps=FPS, codec="libx264", audio_codec="aac",
        bitrate="2200k", audio_bitrate="128k",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        threads=4, logger="bar",
    )
    print("saved:", out, "| duration =", round(final.duration, 1), "s")

if __name__ == "__main__":
    main()
