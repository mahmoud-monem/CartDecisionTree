import pandas as pd
from model import CartDecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(files: list) -> pd.DataFrame:
    try:
        data_frames = [pd.read_csv(f) for f in files]
    except FileExistsError as e:
        print(e)
    data = pd.concat(data_frames, ignore_index=True)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    columns = data.columns[0].split(';')
    newData = pd.DataFrame(columns=columns)

    for idx in range(len(data)):
        row = data.iloc[idx][0].split(";")
        # print(columns)
        # print(row)
        newData.loc[len(newData)] = row
    return newData

def main():
    print("Reading data.......")
    data = read_data(["cardio_train.csv"])
    print("Done!")
    print("Cleaning data......")
    data = clean_data(data)
    print("Done!")
    data.to_excel("cardio_train.csv", index=False)
    model = CartDecisionTree(data, "cardio")
    print("Training model.....")
    # model.fit()
    print("Done!")


if __name__ == "__main__":
    main()
