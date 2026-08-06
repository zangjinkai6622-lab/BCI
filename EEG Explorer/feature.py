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
    for i, item in enumerate(windows):
        print(
            i,
            item["event"],
            item["window"].shape
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
        sample = {}

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
            for key, value in band_power.items():
                sample[f"{channel}_band_power_{key}"] = value

            # Hjorth
            hjorth = analyser.get_hjorth(window, channel)
            for key, value in hjorth.items():
                sample[f"{channel}_hjorth_{key}"] = value
            # Entropy
            entropy = analyser.get_entropy(window, channel)
            for key, value in entropy.items():
                sample[f"{channel}_entropy_{key}"] = value

        # Label
        sample["label"] = label.event_to_label(
            run,
            event
        )
        features.append(sample)
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.astype(float)
    return feature_df