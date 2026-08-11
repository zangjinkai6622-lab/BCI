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
    sfreq = raw.info["sfreq"]          # 采用数据自带的实际采样率
    n_samples = len(df)
    strategy = config.WINDOW_STRATEGY
    windows = []

    skipped_counts = {"skip_code": 0, "unlabeled": 0}
    for idx, (onset, duration, event) in enumerate(zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description
    )):
        # ---- 非 trial 事件 & 无标签事件 过滤（BCI IV 2a 的数字码） ----
        # event 可能是 int/float（BCI IV 2a）或 str（旧数据 T0/T1/T2）
        try:
            ev_code = int(event)
        except (ValueError, TypeError):
            ev_code = None
        if ev_code is not None:
            if ev_code in config.SKIP_EVENT_CODES:
                skipped_counts["skip_code"] += 1
                continue
            if ev_code in config.BCI2A_UNLABELED_CODES:
                skipped_counts["unlabeled"] += 1
                continue

        # ---- 按策略显式计算时间窗 (秒) ----
        if strategy == "annotation":
            if duration < config.ANNOT_MIN_DURATION_SEC:
                print(f"[split_windows] 跳过 idx={idx} event={event}: "
                      f"duration={duration:.3f}s < ANNOT_MIN_DURATION_SEC={config.ANNOT_MIN_DURATION_SEC}s")
                continue
            t_start = onset + config.ANNOT_PAD_START_SEC
            t_end   = onset + duration + config.ANNOT_PAD_END_SEC

        elif strategy == "fixed_offset":
            t_start = onset + config.FIXED_WINDOW_TMIN
            t_end   = onset + config.FIXED_WINDOW_TMAX

        else:
            raise ValueError(
                f"未知 WINDOW_STRATEGY='{strategy}'，"
                f"可选值: 'annotation' | 'fixed_offset'"
            )

        # ---- 转样本索引 + 边界 clamp ----
        start = max(0, int(round(t_start * sfreq)))
        end   = min(n_samples, int(round(t_end * sfreq)))

        # ---- 最小窗口长度 & 空窗口跳过 ----
        if end - start < config.MIN_WINDOW_SAMPLES:
            print(f"[split_windows] 跳过 idx={idx} event={event}: "
                  f"窗口过短 samples={end - start} (需要 >= {config.MIN_WINDOW_SAMPLES}), "
                  f"t=[{t_start:.2f},{t_end:.2f}]s -> idx=[{start},{end}]")
            continue

        window = df.iloc[start:end]
        if window.empty:
            print(f"[split_windows] 跳过 idx={idx} event={event}: window 为空 DataFrame")
            continue

        windows.append({
            "event": event,
            "window": window
        })

    for i, item in enumerate(windows):
        print(i, item["event"], item["window"].shape)
    print(f"[split_windows] 策略={strategy}, 采样率={sfreq}Hz, "
          f"数据总行数={n_samples}, 成功切出窗口数={len(windows)} "
          f"(过滤掉非 trial 事件数={skipped_counts['skip_code']}, "
          f"无标签评估事件数={skipped_counts['unlabeled']})")
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

        # Label
        sample["label"] = label.event_to_label(
            run,
            event
        )
        features.append(sample)
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.astype(float)
    return feature_df