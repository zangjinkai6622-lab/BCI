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