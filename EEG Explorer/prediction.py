import config



def save_prediction_md(file_name:str,model_name:str,prediction:str):
    save_dir = config.PREDICTION_DIR
    save_dir.mkdir(parents=True,exist_ok=True)
    save_path=save_dir/"prediction.md"
    with open(save_path,"w",encoding="utf8") as f:
        f.write("# Prediction Report\n\n")
        f.write("## File\n")
        f.write(f"{file_name}\n\n")
        f.write("## Model\n")
        f.write(f"{model_name}\n\n")
        f.write("## Prediction\n")
        f.write(f"{prediction}\n")

def save_prediction_csv(results:list):

    df = pd.DataFrame(results)

    save_path = config.PREDICTION_DIR / "prediction.csv"

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )