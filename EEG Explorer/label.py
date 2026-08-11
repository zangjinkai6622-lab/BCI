import config

LABEL_NAME = {
    0: "rest",
    1: "left_hand",
    2: "right_hand",
    3: "both_hands",
    4: "both_feet"
}
def decode_label(label):
    return LABEL_NAME.get(label, "unknown")

def event_to_label(run, event):
    # ---- 分支 A：BCI IV 2a 风格的整数事件码（事件本身即类别，无需 run） ----
    try:
        ev_code = int(event)
        # 769~772 → 直接查映射表
        if ev_code in config.BCI2A_EVENT_TO_LABEL:
            return config.BCI2A_EVENT_TO_LABEL[ev_code]
        # 768 / 783 → 评估集无标签，抛错提示（正常会在 split_windows 阶段被过滤）
        if ev_code in config.BCI2A_UNLABELED_CODES:
            raise ValueError(
                f"BCI IV 2a 事件 {ev_code} 无真值标签（评估集 E 文件）。"
                f"训练请使用 T 文件（AxxT.gdf）。"
            )
    except (ValueError, TypeError):
        pass  # event 不是整数，走旧数据分支

    # ---- 分支 B：旧数据风格（T0/T1/T2 字符串 + run 组合判断） ----
    if event == "T0":
        return 0
    # Left / Right
    if run in [3, 4, 7, 8, 11, 12]:
        if event == "T1":
            return 1
        if event == "T2":
            return 2
    # Both Hands / Both Feet
    if run in [5, 6, 9, 10, 13, 14]:
        if event == "T1":
            return 3
        if event == "T2":
            return 4
    raise ValueError(f"Unknown event mapping: run={run}, event={event}")