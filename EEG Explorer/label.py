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

# BCI IV 2a 真值标签白名单（其他一律返回 None，不报错不降级）
#   769 -> 1 (left_hand)
#   770 -> 2 (right_hand)
#   771 -> 4 (both_feet)
#   772 / 768 / 783 / 其他 -> None（由调用方决定是否跳过该窗口）
_BCI2A_LABEL_MAP = {
    769: 1,
    770: 2,
    771: 4,
}

def event_to_label(run, event):
    try:
        ev_code = int(event)
    except (ValueError, TypeError):
        # event 不是整数（如旧数据的 "T0"/"T1"/"T2"）→ 暂不支持，返回 None
        return None
    # 白名单命中 → 返回对应标签；其他一切 → None
    return _BCI2A_LABEL_MAP.get(ev_code, None)