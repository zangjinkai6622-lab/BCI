import pandas as pd
import numpy as np
import re
import config
import analyser
import label

def get_run(file_path: str):
    match = re.search(r"R(\d+)", file_path)
    if match:
        return int(match.group(1))
    return None
def split_windows(raw, df: pd.DataFrame):

    windows = []
    for onset, duration, event in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description
    ):
        start = int(onset * config.SAMPLING_RATE)
        end = int((onset + duration) * config.SAMPLING_RATE)
        window = df.iloc[start:end]
        windows.append({
            "event": event,
            "window": window
        })
    return windows

def create_feature_dataframe(
    windows: list,
    channels: list,
    run: int
):
    features = []
    # =====【修改3-新增MI特征6】定义MI关键通道对（左右运动皮层对称通道），用于计算不对称特征
    # Motor Imagery 经典：左手/右手想象时 C3/C4 出现对侧 ERD
    asym_channel_pairs = [
        ("C3", "C4"),   # 最核心的左右运动皮层
        ("C1", "C2"),   # C3/C4内侧
        ("C5", "C6"),   # C3/C4外侧
        ("Cp3", "Cp4"), # 中央顶区
        ("Cp1", "Cp2"),
        ("Cp5", "Cp6"),
        ("Fc3", "Fc4"), # 额中央区
        ("Fc1", "Fc2"),
        ("Fc5", "Fc6"),
        ("P3",  "P4"),  # 顶叶区
        ("F3",  "F4"),  # 额叶区
    ]
    # 把上面成对通道里实际存在的挑出来
    channel_set = set(channels)
    valid_pairs = [(l, r) for (l, r) in asym_channel_pairs if l in channel_set and r in channel_set]

    for item in windows:
        window = item["window"]
        event = item["event"]
        sample = {}
        # =====【修改3-新增MI特征】临时保存每个通道的band_power，用于后面跨通道不对称计算
        per_channel_band_power = {}

        # Time Domain
        time_features = analyser.get_time_domain_features(window)

        for ch, feature in time_features.items():
            for key, value in feature.items():
                sample[f"{ch}_time_domain_{key}"] = value
        # Frequency Features
        for channel in channels:
            # FFT
            fft_result = analyser.get_fft(
                window,
                channel,
                config.SAMPLING_RATE
            )
            freq = fft_result["frequency"]
            amp = fft_result["amplitude"]
            mask = (freq >= 1) & (freq <= 40)
            peak_freq = freq[mask][np.argmax(amp[mask])]
            sample[f"{channel}_peak_frequency"] = peak_freq
            # PSD Band Power
            psd_result = analyser.get_psd(
                window,
                channel,
                config.SAMPLING_RATE
            )
            band_power = analyser.get_band_power(
                psd_result,
                config.bands
            )
            per_channel_band_power[channel] = band_power  # =====【修改3-新增MI特征6】暂存用于不对称计算
            for key, value in band_power.items():
                sample[f"{channel}_band_power_{key}"] = value

            # =====【修改3-新增MI特征1】相对频带功率（每通道）
            rel_bp = analyser.get_relative_band_power(band_power)
            for key, value in rel_bp.items():
                sample[f"{channel}_{key}"] = value

            # =====【修改3-新增MI特征2】频带比值（每通道）
            ratios = analyser.get_band_ratios(band_power)
            for key, value in ratios.items():
                sample[f"{channel}_{key}"] = value

            # =====【修改3-新增MI特征3】ERD/ERS前后段功率变化特征（每通道）
            erd_ers = analyser.get_erd_ers(window, channel, config.SAMPLING_RATE, config.bands)
            for key, value in erd_ers.items():
                sample[f"{channel}_{key}"] = value

            # =====【修改3-新增MI特征4】频谱形状特征（质心、边缘频率、带宽）
            shape = analyser.get_spectral_shape(psd_result)
            for key, value in shape.items():
                sample[f"{channel}_{key}"] = value

            # Hjorth
            hjorth = analyser.get_hjorth(window, channel)
            for key, value in hjorth.items():
                sample[f"{channel}_hjorth_{key}"] = value
            # Entropy
            entropy = analyser.get_entropy(window, channel)
            for key, value in entropy.items():
                sample[f"{channel}_entropy_{key}"] = value

        # =====【修改3-新增MI特征5】跨通道不对称特征（MI核心！）
        for (ch_left, ch_right) in valid_pairs:
            bp_left  = per_channel_band_power[ch_left]
            bp_right = per_channel_band_power[ch_right]
            asym_feats = analyser.get_channel_asymmetry(bp_left, bp_right, ch_left, ch_right)
            for k, v in asym_feats.items():
                sample[k] = v

        # =====【修改3-新增MI特征7】Cz（中央中线）与左右运动区的差/比，用于双脚/双手MI的区分
        if "Cz" in channel_set:
            bp_cz = per_channel_band_power.get("Cz", {})
            if "C3" in channel_set:
                sample.update(analyser.get_channel_asymmetry(bp_cz, per_channel_band_power["C3"], "Cz", "C3"))
            if "C4" in channel_set:
                sample.update(analyser.get_channel_asymmetry(bp_cz, per_channel_band_power["C4"], "Cz", "C4"))
            if "Cpz" in channel_set and "Cz" in channel_set:
                sample.update(analyser.get_channel_asymmetry(bp_cz, per_channel_band_power["Cz"], "Cpz", "Cz"))

        # Label
        sample["label"] = label.event_to_label(
            run,
            event
        )
        features.append(sample)
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.astype(float)
    return feature_df