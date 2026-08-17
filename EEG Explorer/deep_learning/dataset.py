import feature
import reader

def get_Dataset(file_path: str):
    raw, df = reader.read_gdf(file_path)
    dataset=feature.split_windows(raw, df)