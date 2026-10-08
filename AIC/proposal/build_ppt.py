# -*- coding: utf-8 -*-
"""答辩 PPT 生成：与技术报告同一数据源（results/*.csv、figures/*.png）。

输出：proposal/答辩PPT_机械臂意念控制的自适应脑电解码算法.pptx（16:9）
"""
import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE / "figures"

# Luxury Academic：暖白底 + 深蓝绿主色 + 青绿强调，沉稳有质感
BG = RGBColor(0xFC, 0xFB, 0xF7)        # 暖白背景（比纯白柔和）
INK = RGBColor(0x0F, 0x17, 0x2A)        # 深石墨正文
NAVY = INK
BLUE = RGBColor(0x0F, 0x76, 0x6E)       # 深蓝绿主色：标题/强调
TEAL = RGBColor(0x14, 0xB8, 0xA6)       # 青绿：关键数字/亮点
ACCENT = RGBColor(0xB4, 0x53, 0x0E)     # 深琥珀：诚实结论/警示
LIGHT = RGBColor(0xE6, 0xF2, 0xF0)      # 极淡青绿：卡片/表头底
LIGHT2 = RGBColor(0xF5, 0xF0, 0xEB)     # 极淡暖灰：对照组卡片
DARK = INK
GRAY = RGBColor(0x47, 0x55, 0x69)       # 灰：注释/次要
LINE = RGBColor(0xCB, 0xD5, 0xD1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = ACCENT
GREEN = TEAL

FONT = "微软雅黑"
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def _set_ea(run, name=FONT):
    """安全设置 run 的东亚字体（避免 rPr/rFonts 不存在时报错）。"""
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("a:rFonts"))
    if rFonts is None:
        rFonts = etree.SubElement(rPr, qn("a:rFonts"))
    rFonts.set("eastAsia", name)

# ------------------------------------------------------------ 数据
def read_csv(name):
    with open(RES / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

cmp = {r["model"]: r for r in read_csv("comparison.csv")}
abl = {r["ablation"]: r for r in read_csv("ablation.csv")}
los = {r["model"]: r for r in read_csv("cross_subject.csv")}
# 2b 迁移学习结果（2b 预训练 -> 2a 微调）
tl = {r["model"]: r for r in read_csv("transfer_loso.csv")}
transfer_acc = float(next(iter(tl.values()))["accuracy_mean"]) * 100
transfer_std = float(next(iter(tl.values()))["accuracy_std"]) * 100
flow = {r["metric"]: r["value"] for r in read_csv("calibration_flow.csv")}

# ------------------------------------------------------------ 基元
def slide():
    s = prs.slides.add_slide(BLANK)
    # 暖白背景铺满
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    # 把背景移到最底层
    spTree = s.shapes._spTree
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)
    return s

def rect(s, x, y, w, h, color, line=None):
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp

def rrect(s, x, y, w, h, color, line=None, lw=1.0):
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    return sp

def text(s, x, y, w, h, content, size=18, color=DARK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, font=FONT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(1)
    lines = content.split("\n") if isinstance(content, str) else content
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color; r.font.name = font
        _set_ea(r, font)
    return tb

def title_bar(s, t, tag=""):
    # 左侧深蓝绿色块
    rect(s, 0, 0, 0.18, 1.12, BLUE)
    # 顶部细线
    rect(s, 0.18, 0, 13.15, 0.04, LIGHT)
    # 标题
    text(s, 0.5, 0.2, 10.5, 0.82, t, size=26, color=INK, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
    # 标题下装饰线（左蓝右淡）
    rect(s, 0.5, 1.02, 2.2, 0.035, BLUE)
    rect(s, 2.7, 1.035, 10.2, 0.012, LINE)
    if tag:
        # 标签用青绿圆角小标签
        rrect(s, 10.8, 0.32, 2.3, 0.5, LIGHT, line=None)
        text(s, 10.8, 0.32, 2.3, 0.5, tag, size=11, color=BLUE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, bold=True)

def footer(s, n):
    text(s, 0.4, 7.08, 8, 0.35,
         "机械臂意念控制的自适应脑电解码算法", size=10, color=GRAY)
    text(s, 12.3, 7.08, 0.8, 0.35, str(n), size=11, color=GRAY, align=PP_ALIGN.RIGHT)

def pic(s, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)

def bullet(s, x, y, w, h, items, size=16, gap=6, color=DARK, mk="▪ ", mkcolor=BLUE):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r1 = p.add_run(); r1.text = mk; r1.font.size = Pt(size); r1.font.color.rgb = mkcolor
        r1.font.name = FONT; r1.font.bold = True
        r2 = p.add_run(); r2.text = it; r2.font.size = Pt(size); r2.font.color.rgb = color
        r2.font.name = FONT
        _set_ea(r2)
    return tb

def pct(v, sd=None):
    s = f"{float(v)*100:.2f}%"
    return s + (f" ± {float(sd)*100:.2f}" if sd is not None else "")

def _cell_edge(cell, edge, spec):
    """设置单元格单边边框。edge ∈ L/R/T/B；spec=None 无线，或 (磅值, RGBColor)。"""
    tag = {"L": "a:lnL", "R": "a:lnR", "T": "a:lnT", "B": "a:lnB"}[edge]
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn(tag))
    if old is not None:
        tcPr.remove(old)
    el = etree.Element(qn(tag))
    if spec is None:
        etree.SubElement(el, qn("a:noFill"))
    else:
        ptv, col = spec
        el.set("w", str(int(ptv * 12700)))
        el.set("cap", "flat"); el.set("cmp", "sng"); el.set("algn", "ctr")
        f = etree.SubElement(el, qn("a:solidFill"))
        c = etree.SubElement(f, qn("a:srgbClr"))
        c.set("val", str(col))
    # 边框元素在 schema 中必须位于填充元素之前
    anchor = None
    for ftag in ("a:noFill", "a:solidFill", "a:gradFill", "a:blipFill",
                 "a:pattFill", "a:grpFill"):
        e = tcPr.find(qn(ftag))
        if e is not None:
            anchor = e
            break
    if anchor is not None:
        anchor.addprevious(el)
    else:
        tcPr.append(el)

def ppt_table(s, x, y, w, header, rows, col_w=None, fs=13, hfs=13,
              row_h=0.42, highlight_row=None):
    nr, nc = len(rows) + 1, len(header)
    gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                            Inches(row_h * nr)).table
    # 关闭主题表格样式（蓝白条纹/首行强调）
    tblPr = gt._tbl.tblPr
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    for ts in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(ts)
    if col_w:
        total = sum(col_w)
        for j, cw in enumerate(col_w):
            gt.columns[j].width = Emu(int(Inches(w) * cw / total))
    for j, htxt in enumerate(header):
        c = gt.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = LIGHT
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_top = c.margin_bottom = Pt(1)
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = htxt; r.font.size = Pt(hfs); r.font.bold = True
        r.font.color.rgb = BLUE; r.font.name = FONT
        _set_ea(r)
        _cell_edge(c, "L", None); _cell_edge(c, "R", None)
        _cell_edge(c, "T", (1.5, INK)); _cell_edge(c, "B", (1.0, INK))
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        hl = highlight_row is not None and i == highlight_row
        for j, v in enumerate(row):
            c = gt.cell(i + 1, j)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if hl else WHITE
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = c.margin_bottom = Pt(1)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(v); r.font.size = Pt(fs)
            r.font.color.rgb = INK; r.font.name = FONT
            if hl:
                r.font.bold = True
            _set_ea(r)
            _cell_edge(c, "L", None); _cell_edge(c, "R", None)
            _cell_edge(c, "T", None)
            _cell_edge(c, "B", (1.5, INK) if last else None)
    return gt

n = 0
def pno():
    global n; n += 1; return n

# ============================================================ 1 封面
s = slide()
# 左侧竖向装饰
rect(s, 0, 0, 0.6, 7.5, BLUE)
rect(s, 0.6, 0, 0.06, 7.5, TEAL)
# 顶部细条
rect(s, 0.66, 0, 12.67, 0.05, LIGHT)
# 底部信息区
rect(s, 0.66, 5.6, 12.67, 1.9, LIGHT)
rect(s, 0.66, 5.6, 12.67, 0.03, BLUE)
# 赛事名
text(s, 1.0, 1.0, 11.8, 0.5,
     "2026 全球校园人工智能算法精英大赛 · 算法创新赛（AI＋学科交叉）",
     size=14, color=GRAY, align=PP_ALIGN.CENTER)
# 主标题
text(s, 1.0, 2.1, 11.8, 1.7,
     "机械臂意念控制的\n自适应脑电解码算法", size=42, color=INK, bold=True,
     align=PP_ALIGN.CENTER)
# 装饰线
rect(s, 5.0, 4.15, 3.3, 0.04, BLUE)
rect(s, 5.6, 4.24, 2.1, 0.015, TEAL)
# 副标题
text(s, 1.0, 4.45, 11.8, 0.6,
     "运动想象脑电解码 · 跨被试免校准 · 三自由度辅助机械臂",
     size=18, color=BLUE, align=PP_ALIGN.CENTER, bold=True)
# 底部信息
text(s, 1.0, 5.85, 11.8, 1.2,
     "团队：____________      团队编号：____________\n公开基准 BCI Competition IV-2a ｜ 9 被试 × 2 场次 × 3888 试验",
     size=13, color=GRAY, align=PP_ALIGN.CENTER)

# ============================================================ 2 目录
s = slide(); title_bar(s, "汇报提纲")
items = ["项目背景与核心目标", "需求分析与学科界定", "系统架构与数据平台",
         "STFA-Net 与 MAG 门控创新", "跨被试免校准方法",
         "实验结果：对比 / 消融 / LOSO", "校准流程量化对比",
         "总结、展望与演示"]
for i, it in enumerate(items):
    col = i % 2; row = i // 2
    x = 1.0 + col * 6.1; y = 1.6 + row * 1.25
    text(s, x, y, 0.75, 0.6, f"{i+1:02d}", size=20, color=BLUE, bold=True,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.8, y, 5.0, 0.6, it, size=18, color=INK,
         anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x + 0.8, y + 0.62, 4.7, 0.012, LINE)
footer(s, pno())

# ============================================================ 3 背景
s = slide(); title_bar(s, "一、项目背景与痛点", "神经工程 / 康复工程")
text(s, 0.45, 1.25, 12.4, 0.7,
     "通过解析大脑感觉运动区 EEG，帮助 ALS、高位截瘫、脑卒中患者用意念控制辅助机械臂",
     size=17, color=BLUE, bold=True)
cards = [("样本稀缺", "每用户仅约 200 次\n4s 有标签试验", "深度模型易过拟合"),
         ("校准耗时", "新用户需 20–30 分钟\n有标签标定", "占用康复黄金时间"),
         ("跨被试漂移", "脑形态/阻抗差异大\n换人即可能失效", "实训难标准化")]
for i, (t1, t2, t3) in enumerate(cards):
    x = 0.45 + i * 4.35
    rrect(s, x, 2.2, 4.0, 3.6, LIGHT, line=LINE)
    text(s, x + 0.25, 2.32, 3.5, 0.7, t1, size=19, color=BLUE, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x + 1.3, 3.12, 1.4, 0.02, LINE)
    text(s, x + 0.25, 3.4, 3.5, 1.3, t2, size=16, color=INK,
         align=PP_ALIGN.CENTER)
    text(s, x + 0.25, 4.8, 3.5, 0.8, t3, size=14, color=GRAY,
         align=PP_ALIGN.CENTER)
text(s, 0.45, 6.1, 12.4, 0.6,
     "动作映射：左手 → 左移      右手 → 右移      双脚 → 抓取",
     size=17, color=DARK, bold=True, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 4 需求分析
s = slide(); title_bar(s, "二、需求分析与学科界定")
rrect(s, 0.45, 1.3, 5.9, 2.0, LIGHT)
text(s, 0.7, 1.5, 5.4, 0.5, "学科归属", size=17, color=BLUE, bold=True)
bullet(s, 0.7, 2.05, 5.5, 1.2,
       ["生物医学工程 · 神经工程 / 康复工程", "环节：神经科学实验实训 + MI-BCI 科研",
        "目标：实验效率 + 可及性 + 标准化"], size=14, gap=4)
rrect(s, 6.95, 1.3, 5.9, 2.0, LIGHT)
text(s, 7.2, 1.5, 5.4, 0.5, "AI 赋能切入点", size=17, color=BLUE, bold=True)
bullet(s, 7.2, 2.05, 5.5, 1.2,
       ["小样本自适应建模（MAG 门控）", "无标签跨被试对齐（EA）",
        "实时推理 + 机械臂闭环控制"], size=14, gap=4)
ppt_table(s, 0.45, 3.7, 12.4,
          ["学科需求", "AI 技术", "模型模块"],
          [["样本少、防过拟合", "零初始化门控 + 正则", "MAG"],
           ["空间模式判别", "深度空间卷积", "Spatial Conv"],
           ["多节律时频结构", "多尺度时间 + STFT", "Temporal / Frequency"],
           ["新用户免校准", "无标签 EA + 域对抗", "EA / GRL"]],
          col_w=[1, 1.15, 1.1], fs=14, row_h=0.55)
footer(s, pno())

# ============================================================ 5 架构
s = slide(); title_bar(s, "三、系统总体架构")
pic(s, FIG / "fig1_system_architecture.png", 0.55, 1.5, w=12.2)
text(s, 0.55, 6.75, 12.2, 0.4,
     "信号采集（22 导）→ 预处理（1–40Hz 带通、50Hz 陷波）→ STFA-Net 解码 → 三自由度机械臂",
     size=13, color=GRAY, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 6 STFA
s = slide(); title_bar(s, "四、STFA-Net 时空频自适应网络")
pic(s, FIG / "fig2_stfa_network.png", 0.5, 1.35, w=8.1)
rrect(s, 8.95, 1.4, 3.95, 5.3, LIGHT)
text(s, 9.2, 1.6, 3.5, 0.5, "四条并行分支", size=17, color=BLUE, bold=True)
bullet(s, 9.2, 2.2, 3.6, 4.3,
       ["多尺度时间卷积\n核 16 / 32 / 64（0.32–1.28s）",
        "深度空间卷积\n跨 22 电极，groups=16",
        "STFT 频域分支\n1–40Hz，3×3 卷积",
        "通道注意力\nSE 式可靠度加权",
        "融合：Concat + Dropout + FC"], size=14, gap=10)
footer(s, pno())

# ============================================================ 7 MAG
s = slide(); title_bar(s, "五、核心创新：模块可靠性自适应门控 MAG")
pic(s, FIG / "fig3_gating_mechanism.png", 0.5, 1.45, w=7.4)
rrect(s, 8.2, 1.5, 4.7, 4.9, LIGHT)
text(s, 8.45, 1.7, 4.2, 0.5, "设计要点", size=17, color=BLUE, bold=True)
bullet(s, 8.45, 2.25, 4.3, 3.0,
       ["残差形式：x · (1 + αw)",
        "α 零初始化：开训等价恒等映射",
        "α 可导出：训练后门控强度即裁决",
        "数据梯度自动取舍，替代先验堆叠"], size=15, gap=10)
text(s, 0.5, 6.35, 12.3, 0.7,
     "实测：通道 −0.09 / 时间 −0.40 / 特征 −0.34 / 频域 −0.14（5 seeds），"
     "弱模块被自适应抑制，与消融结果互证",
     size=14, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 8 跨被试
s = slide(); title_bar(s, "六、跨被试免校准泛化")
pic(s, FIG / "fig4_cross_subject.png", 0.5, 1.4, w=7.6)
rrect(s, 8.45, 1.5, 4.45, 5.0, LIGHT)
text(s, 8.7, 1.7, 4, 0.5, "两级机制", size=17, color=BLUE, bold=True)
bullet(s, 8.7, 2.3, 4.0, 2.2,
       ["EA 欧氏对齐：测试端\n仅用自身无标签信号，22.5ms",
        "GRL 梯度反转：被试\n鉴别器反向编码域不变特征"], size=14, gap=12)
text(s, 8.7, 4.7, 4, 0.5, "协议", size=17, color=BLUE, bold=True)
bullet(s, 8.7, 5.25, 4.0, 1.2,
       ["LOSO 留一被试 · 9 折", "每折 × 5 seeds 重复",
        "对齐/划分全折统一，公平对照"], size=14, gap=4)
footer(s, pno())

# ============================================================ 9 被试内结果
s = slide(); title_bar(s, "七、实验结果（1）被试内对比", "5 seeds · 3888 试验")
rows = []
order = [("RF", "RF（手工特征）"), ("SVM", "SVM（手工特征）"),
         ("LR", "LR（手工特征）"), ("eegnet", "EEGNet"),
         ("stfa", "STFA-Net"), ("stfa_dg", "STFA-DGNet")]
hi_idx = None
for i, (k, lb) in enumerate(order):
    r = cmp[k]
    if "params" in r and r["params"]:
        rows.append([lb, r["params"], pct(r["accuracy_mean"], r["accuracy_std"]),
                     pct(r["macro_f1_mean"], r["macro_f1_std"]),
                     f"{float(r['inference_ms']):.3f}"])
    else:
        rows.append([lb, "—", pct(r["accuracy_mean"]),
                     pct(r["macro_f1_mean"]), f"{float(r['inference_ms']):.3f}"])
    if k == "stfa": hi_idx = i
ppt_table(s, 0.45, 1.55, 12.4,
          ["模型", "参数量", "Accuracy", "Macro-F1", "推理 ms/样本"],
          rows, col_w=[1.5, 0.9, 1.2, 1.2, 1.15], fs=14, row_h=0.62,
          highlight_row=hi_idx)
text(s, 0.45, 6.2, 12.4, 0.6,
     f"STFA-Net 较 EEGNet +{(float(cmp['stfa']['accuracy_mean'])-float(cmp['eegnet']['accuracy_mean']))*100:.2f}pp，"
     f"较传统 ML +{float(cmp['stfa']['accuracy_mean'])*100-max(float(cmp[k]['accuracy_mean'])*100 for k in ('RF','SVM','LR')):.1f}pp 以上",
     size=16, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 10 消融
s = slide(); title_bar(s, "七、实验结果（2）消融实验", "逐模块移除 · 5 seeds")
abl_order = [("stfa_full", "完整模型"), ("wo_multiscale_time", "去多尺度时间"),
             ("wo_spatial", "去空间卷积"), ("wo_attention", "去注意力"),
             ("wo_frequency", "去频域")]
base = float(abl["stfa_full"]["accuracy_mean"])
rows, sp_idx = [], None
for i, (k, lb) in enumerate(abl_order):
    r = abl[k]
    d = "—" if k == "stfa_full" else f"{(float(r['accuracy_mean'])-base)*100:+.2f}"
    rows.append([lb, r["params"], pct(r["accuracy_mean"], r["accuracy_std"]), d,
                 pct(r["macro_f1_mean"], r["macro_f1_std"])])
    if k == "wo_spatial": sp_idx = i
ppt_table(s, 0.45, 1.55, 12.4,
          ["配置", "参数量", "Accuracy", "Δ pp", "Macro-F1"],
          rows, col_w=[1.4, 0.9, 1.3, 0.8, 1.3], fs=14, row_h=0.6,
          highlight_row=sp_idx)
text(s, 0.45, 6.15, 12.4, 0.85,
     "空间卷积为硬依赖（−17.71pp、F1 −21.83pp）；多尺度时间弱贡献（−1.00pp）；\n"
     "注意力/频域移除反升，与 MAG 门控 α 实测为负相互印证",
     size=15, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 11 LOSO
s = slide(); title_bar(s, "七、实验结果（3）跨被试 LOSO", "9 折 × 5 seeds")
rows, hi = [], None
for i, (k, lb) in enumerate([("eegnet", "EEGNet"),
                             ("stfa", "STFA-Net（EA，无对抗）"),
                             ("stfa_dg", "STFA-DGNet（EA+GRL）"),
                             ("transfer", "STFA-Net（2b 预训练迁移）")]):
    if k == "transfer":
        rows.append([lb, f"{transfer_acc:.2f} ± {transfer_std:.2f}%",
                     "—", "—"])
        hi = i
    else:
        r = los[k]
        rows.append([lb, pct(r["accuracy_mean"], r["accuracy_std"]),
                     pct(r["macro_f1_mean"], r["macro_f1_std"]),
                     f"{float(r['inference_ms']):.3f}"])
ppt_table(s, 0.45, 1.55, 12.4,
          ["模型", "LOSO Accuracy", "Macro-F1", "推理 ms/样本"],
          rows, col_w=[1.8, 1.2, 1.2, 1.1], fs=14, row_h=0.55)
text(s, 0.45, 4.3, 12.4, 1.9,
     "诚实结论：9 被试规模下 GRL 增益 −0.02pp；2b 跨数据集预训练迁移使 LOSO 由 51.57% 提升至 52.48%（+0.91pp）；\n"
     "折间 std ≈ 15pp，跨被试主要瓶颈是被试间分布差异而非模型容量；\n"
     "2b 独立基准验证：被试内 76.90%、LOSO 66.51%，多数据集泛化能力得到佐证。",
     size=15, color=DARK, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 12 校准对比
s = slide(); title_bar(s, "八、新用户上线流程量化对比", "本机实测")
# 传统方案：暖灰底灰框
rrect(s, 0.45, 1.4, 6.0, 4.9, LIGHT2, line=LINE)
text(s, 0.45, 1.5, 6.0, 0.7, "传统有标签校准", size=18, color=GRAY, bold=True,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 1.45, 2.28, 4.0, 0.018, LINE)
bullet(s, 0.8, 2.55, 5.4, 3.4,
       ["标签试验 120 个（3 类 × 40）",
        f"范式采集约 {float(flow['labeled_acquisition_minutes_120']):.0f} 分钟（9.30 s/试验）",
        f"个性化训练 CPU 约 {float(flow['cpu_train_seconds_stfa_120']):.0f} s",
        "STFA 56.79 ± 13.05%"], size=16, gap=14,
       mkcolor=GRAY)
# 本作品：淡蓝底蓝框强调
rrect(s, 6.9, 1.4, 6.0, 4.9, LIGHT, line=BLUE, lw=1.5)
text(s, 6.9, 1.5, 6.0, 0.7, "本作品免校准", size=18, color=BLUE, bold=True,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 7.9, 2.28, 4.0, 0.022, BLUE)
bullet(s, 7.25, 2.55, 5.4, 3.4,
       ["标签试验 0 个", "无范式采集，模型跨用户共享",
        f"EA 对齐 {float(flow['ea_estimate_seconds_K216'])*1000:.1f} ms（216 无标签试验）",
        f"LOSO {transfer_acc:.2f} ± {transfer_std:.2f}%（2b 迁移）"], size=16, gap=14, mkcolor=BLUE)
text(s, 0.45, 6.45, 12.4, 0.6,
     f"约 19 分钟有标签采集 ⇄ {(56.79 - transfer_acc):.2f} 个百分点：效率收益可量化、可核对",
     size=17, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
footer(s, pno())

# ============================================================ 13 应用与价值
s = slide(); title_bar(s, "九、应用演示与价值")
bullet(s, 0.7, 1.5, 5.7, 4.5,
       ["新用户佩戴头帽，秒级 EA 对齐",
        "想象肢体运动 → STFA 实时推理",
        "三指令驱动机械臂闭环动作",
        "展示决策脑区与置信度，可示教",
        "CPU 即可运行（≈3.1 ms/片段）"], size=17, gap=14)
rrect(s, 7.0, 1.5, 5.8, 4.6, LIGHT)
text(s, 7.3, 1.75, 5.2, 0.5, "三类价值（可量化）", size=18, color=BLUE, bold=True)
ppt_table(s, 7.25, 2.5, 5.3,
          ["维度", "指标"],
          [["效率", "取消约 19 min 标定"],
           ["性能", "较 EEGNet +2.76pp"],
           ["可教学", "α + 地形图可视化"]],
          col_w=[0.8, 1.5], fs=15, row_h=0.75)
text(s, 7.25, 5.25, 5.3, 0.7,
     "演示视频同步提交（真实信号回放）", size=14, color=GRAY)
footer(s, pno())

# ============================================================ 14 创新总结
s = slide(); title_bar(s, "十、创新点总结")
inn = [("MAG 零初始化门控", "x·(1+αw)，α 可导出，数据裁决模块"),
       ("免校准推理流程", "EA（无标签、22.5ms）+ GRL 域对抗"),
       ("严谨实验协议", "5 seeds + 消融 + LOSO，负面结果诚实报告"),
       ("轻量化可部署", "13,469 参数，CPU 实时，无需专用硬件")]
for i, (t1, t2) in enumerate(inn):
    col = i % 2; row = i // 2
    x = 0.6 + col * 6.3; y = 1.6 + row * 2.4
    rrect(s, x, y, 5.9, 2.0, WHITE, line=LINE)
    text(s, x + 0.3, y + 0.18, 0.9, 0.7, f"{i+1:02d}", size=22,
         color=LINE, bold=True)
    text(s, x + 1.15, y + 0.22, 4.5, 0.7, t1, size=20, color=BLUE, bold=True)
    text(s, x + 1.15, y + 1.02, 4.5, 0.8, t2, size=15, color=INK)
footer(s, pno())

# ============================================================ 15 展望
s = slide()
rect(s, 0, 0, 13.333, 0.14, BLUE)
rect(s, 0, 5.8, 13.333, 1.7, LIGHT)
rect(s, 0, 5.8, 13.333, 0.025, BLUE)
text(s, 1, 1.5, 11.3, 0.9, "总结与展望", size=34, color=INK, bold=True,
     align=PP_ALIGN.CENTER)
rect(s, 5.17, 2.5, 3.0, 0.03, BLUE)
bullet(s, 3.1, 2.95, 7.6, 2.6,
       ["无监督域适应：缩小被试间分布差异",
        "个性化在线自适应：信号质量条件化门控",
        "真实康复场景试点：ALS/卒中患者验证",
        "跨数据集泛化：IV-2b / PhysioNet"], size=19, gap=14,
       color=INK, mkcolor=BLUE)
text(s, 1, 6.15, 11.3, 0.9, "感谢各位老师！恳请批评指正。", size=26,
     color=BLUE, bold=True, align=PP_ALIGN.CENTER)

import shutil
out = HERE / "答辩PPT_机械臂意念控制的自适应脑电解码算法.pptx"
tmp = HERE / "_tmp_ppt.pptx"
prs.save(str(tmp))
try:
    shutil.move(str(tmp), str(out))
except PermissionError:
    # 如果仍被占用，保留临时文件
    out = tmp
    print("WARN: 原文件被占用，保存为临时文件")
print("saved:", out, "| slides =", len(prs.slides))
