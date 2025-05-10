import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml



#df = pd.read_csv(r"C:\Users\Sande\Desktop\Datasets\medical_insurance.csv")

#test_size = yaml.safe_load(open("params.yaml"))['data_collection']['test_size']

def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params=yaml.safe_load(file)
        return  params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"error loading parameters form {filepath}: {e}")

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return  pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error loading data from {filepath}: {e}")
 


def split_data(df: pd.DataFrame, test_size: float):
        try:
            return train_test_split(df, random_state=42, test_size=test_size)
        except ValueError as e:
            raise ValueError(f"error splitting data: {e}")


#train_data, test_data = train_test_split(df, random_state=42, test_size=test_size)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"error saveing data to {filepath}: {e}")


def main():
    data_filepath= r"C:\Users\Sande\Desktop\Datasets\medical_insurance.csv"
    params_filepath = "params.yaml"
    raw_data_path =  os.path.join("data", "raw")

#data_path = os.path.join("data", "raw")
    try:
        data=load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)

        os.makedirs(raw_data_path)


        save_data(train_data, os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"an error occurred:{e}")

if __name__ == "__main__":
    main()    

