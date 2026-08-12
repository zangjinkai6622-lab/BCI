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
    sfreq = raw.info["sfreq"]
    n_samples = len(df)
    strategy = config.WINDOW_STRATEGY
    windows = []

    skipped_counts = {
        "skip_code": 0,
        "unlabeled": 0,
        "invalid_window": 0
    }

    for idx, (onset, duration, event) in enumerate(
        zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description
        )
    ):

        # --------------------------------------------------
        # 1. 尝试转换事件码
        # --------------------------------------------------
        try:
            ev_code = int(event)
        except (ValueError, TypeError):
            ev_code = None

        # --------------------------------------------------
        # 2. 跳过明确不参与训练的事件
        # --------------------------------------------------
        if ev_code is not None:

            if ev_code in config.SKIP_EVENT_CODES:
                skipped_counts["skip_code"] += 1
                continue

            if ev_code in config.BCI2A_UNLABELED_CODES:
                skipped_counts["unlabeled"] += 1
                continue

        # --------------------------------------------------
        # 3. 获取真实标签
        # --------------------------------------------------
        event_label = label.event_to_label(
            None,
            event
        )

        # BCI IV 2a 白名单之外的事件全部跳过
        if event_label is None:
            skipped_counts["unlabeled"] += 1
            continue

        # --------------------------------------------------
        # 4. 根据窗口策略确定时间范围
        # --------------------------------------------------
        if strategy == "annotation":

            if duration < config.ANNOT_MIN_DURATION_SEC:
                continue

            t_start = (
                onset +
                config.ANNOT_PAD_START_SEC
            )

            t_end = (
                onset +
                duration +
                config.ANNOT_PAD_END_SEC
            )

        elif strategy == "fixed_offset":

            t_start = (
                onset +
                config.FIXED_WINDOW_TMIN
            )

            t_end = (
                onset +
                config.FIXED_WINDOW_TMAX
            )

        else:
            raise ValueError(
                f"未知 WINDOW_STRATEGY='{strategy}'，"
                f"可选值: 'annotation' | 'fixed_offset'"
            )

        # --------------------------------------------------
        # 5. 秒 → sample
        # --------------------------------------------------
        start = max(
            0,
            int(round(t_start * sfreq))
        )

        end = min(
            n_samples,
            int(round(t_end * sfreq))
        )

        # --------------------------------------------------
        # 6. 检查窗口长度
        # --------------------------------------------------
        window_length = end - start

        if window_length < config.MIN_WINDOW_SAMPLES:
            skipped_counts["invalid_window"] += 1
            continue

        window = df.iloc[start:end]

        if window.empty:
            skipped_counts["invalid_window"] += 1
            continue

        # --------------------------------------------------
        # 7. 保存窗口 + 事件 + 标签
        # --------------------------------------------------
        windows.append({
            "event": event,
            "label": event_label,
            "window": window
        })

    # ------------------------------------------------------
    # 输出检查信息
    # ------------------------------------------------------
    for i, item in enumerate(windows):
        print(
            i,
            "event=", item["event"],
            "label=", item["label"],
            "shape=", item["window"].shape
        )

    print(
        f"[split_windows] "
        f"策略={strategy}, "
        f"采样率={sfreq}Hz, "
        f"数据总行数={n_samples}, "
        f"成功切出窗口数={len(windows)}, "
        f"skip_code={skipped_counts['skip_code']}, "
        f"unlabeled={skipped_counts['unlabeled']}, "
        f"invalid_window={skipped_counts['invalid_window']}"
    )

    return windows

def create_feature_dataframe(
    windows: list,
    channels: list,
    run: int
):
    features = []

    for item in windows:
        window = item["window"]
        event = item["event"]
        event_label = item["label"]
        sample = {}

        # Time Domain
        time_features = analyser.get_time_domain_features(window)

        for ch, feature in time_features.items():
            for key, value in feature.items():
                sample[f"{ch}_time_domain_{key}"] = value
        # Frequency Features
        channel_band_power={}
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
            if(channel == "C3" or channel == "C4"):
                channel_band_power[channel]=band_power
            relative_band_power = analyser.get_relative_band_power(band_power)
            for key, value in band_power.items():
                sample[f"{channel}_band_power_{key}"] = value
                # Relative Band Power
                sample[f"{channel}_relative_band_power_{key}"] = relative_band_power[key]
            # Band Ratio
            band_ratio = analyser.get_band_ratio(band_power)
            for key, value in band_ratio.items():
                sample[f"{channel}_band_ratio_{key}"] = value

            # Hjorth
            hjorth = analyser.get_hjorth(window, channel)
            for key, value in hjorth.items():
                sample[f"{channel}_hjorth_{key}"] = value
            # Entropy
            entropy = analyser.get_entropy(window, channel)
            for key, value in entropy.items():
                sample[f"{channel}_entropy_{key}"] = value

    # channel asymmetry
        if "C3" in channel_band_power and "C4" in channel_band_power:
            c3_band_power = channel_band_power["C3"]
            c4_band_power = channel_band_power["C4"]
            asymmetry = analyser.get_channel_asymmetry(
                c3_band_power,
                c4_band_power
            )
            for key, value in asymmetry.items():
                sample[key] = value
        else:
            # 数据集中缺少 C3 或 C4 通道时，asymmetry 特征填 NaN（后续可按需丢弃该列）
            missing = [ch for ch in ("C3", "C4") if ch not in channel_band_power]
            print(f"[create_feature_dataframe] 事件={event}: 缺少通道 {missing}，"
                  f"跳过 channel asymmetry 计算，填 NaN（当前可用通道={sorted(channel_band_power.keys())})")
            sample["alpha_asymmetry"] = np.nan
            sample["beta_asymmetry"] = np.nan

        # Label（直接使用 split_windows 阶段已确定好的 event_label，不做二次映射）
        sample["label"] = event_label
        features.append(sample)
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.astype(float)
    return feature_df