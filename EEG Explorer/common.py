import reader
import feature
import channel

raw, df = reader.read_edf("EEG Explorer/data/S001R14.edf")

channels = channel.get_available_channels(df)

windows = feature.split_windows(raw, df)

print(len(windows))
print(windows[0]["event"])
print(windows[0]["window"].shape)

run = feature.get_run("EEG Explorer/data/S001R14.edf")

feature_df = feature.create_feature_dataframe(
    windows,
    channels,
    run
)

print(feature_df.head())
print(feature_df["label"].value_counts())