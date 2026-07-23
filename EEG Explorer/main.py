import train
import inference


if __name__ == '__main__':
    mode=input("train or predict:")
    if mode=="train":
        train.train_main()
    elif mode=="predict":
        result=inference.predict_file("EEG Explorer/data/S001R003.edf")
        print(result)
