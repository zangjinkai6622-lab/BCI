import os
# label的作用就是标记不同的活动，这也是之后要预测的内容
LABEL_MAP={
    "S001R01.edf":0,
    "S001R02.edf":1,
    "S001R03.edf":2
}

def get_label(file:str):
    filename=os.path.basename(file)
    return LABEL_MAP.get(filename)