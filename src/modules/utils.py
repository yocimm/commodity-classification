import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_data(file_path):
    return pd.read_csv(file_path)


def split_and_save_data(dataset_path, train_path, test_path, test_size=0.2, random_state=42):
    df = load_data(dataset_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Data train disimpan di {train_path}")
    print(f"Data test disimpan di {test_path}")


if __name__ == "__main__":
    dataset_file = r"D:\PIHC\commodity-classification\data\training_data\data_without_nan_resampled.csv"
    train_file = os.path.join("data", "training_data", "train.csv")
    test_file = os.path.join("data", "training_data", "test.csv")
    split_and_save_data(dataset_file, train_file, test_file)
