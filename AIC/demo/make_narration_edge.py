# -*- coding: utf-8 -*-
"""用微软 Edge 神经网络语音（zh-CN-YunxiNeural）重新生成演示视频旁白。

依赖：demo/_pylibs 下的 edge-tts（pip --target 安装）；moviepy 负责 mp3->wav。
输出：audio/<scene>.wav（覆盖旧 Huihui 版本，旧文件备份到 audio/huihui_backup/），
      并打印每段时长供 make_demo_video.py 校准时间轴。
用法：python make_narration_edge.py
"""
import sys
import shutil
import asyncio
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_pylibs"))

import edge_tts
from moviepy import AudioFileClip

VOICE = "zh-CN-YunxiNeural"
# 口语化短句配合自然停顿。不要用过高语速去硬塞进画面，避免机械感。
RATE = "-2%"
AUD = HERE / "audio"
MP3 = HERE / "audio_edge_mp3"
BAK = AUD / "huihui_backup"

SCENES = {
    "s0_title": "这是一套让机械臂听懂意念的脑电解码系统。",
    "s1_background": "当人无法自如行动，脑机接口能把脑中的动作想象变成指令。可每换一位用户，往往都要重新采集、反复校准，模型也容易失准。这正是我们想解决的问题。",
    "s2_architecture": "脑电先做基础清理，再从时间、空间和频率三个角度共同判断。网络会自己调节不同模块的作用。",
    "s3_demo_a": "接下来是完成无标签对齐后的新用户演示。模型没有用过当前用户的标签。四秒信号进入后，系统持续更新三类动作的概率，直到指令稳定。",
    "s3_demo_b": "类别概率稳定后，系统锁定指令：左手左移，右手右移，双脚抓取。想象、解码、执行和反馈，就这样连成闭环。",
    "s4_results": "这组结果来自九名被试、两场次数据。STFA-Net 被试内准确率为百分之六十六点零二，比 EEGNet 高二点七六个百分点；去掉空间信息后，下降最明显。",
    "s5_calib": "传统流程要采集标签。这里，我们只用无标签脑电完成对齐，省去标注和专门校准。",
    "s6_summary": "接下来，我们会把它带到更接近真实康复的场景中，继续验证稳定性和实用性。谢谢观看。",
}


async def _gen(names):
    MP3.mkdir(exist_ok=True)
    for name in names:
        text = SCENES[name]
        out = MP3 / f"{name}.mp3"
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))
                break
            except Exception as e:  # 网络抖动重试
                if attempt == 2:
                    raise
                print(f"retry {name}: {e}")
                await asyncio.sleep(1.5)
        print("mp3:", out.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(SCENES),
                        help="只生成指定片段；用于网络不稳定时分段重试。")
    args = parser.parse_args()
    names = args.only or list(SCENES)
    AUD.mkdir(exist_ok=True)
    BAK.mkdir(exist_ok=True)
    asyncio.run(_gen(names))
    durations = {}
    for name in names:
        mp3 = MP3 / f"{name}.mp3"
        wav = AUD / f"{name}.wav"
        if wav.exists():
            shutil.copy2(wav, BAK / wav.name)   # 备份旧 Huihui 配音
        clip = AudioFileClip(str(mp3))
        clip.write_audiofile(str(wav), fps=22050, nbytes=2,
                             codec="pcm_s16le", logger=None)
        durations[name] = round(clip.duration, 2)
        clip.close()
    print("\n=== narration durations (s) ===")
    for k, v in durations.items():
        print(f"{k}: {v}")
    print("total:", round(sum(durations.values()), 1))


if __name__ == "__main__":
    main()
