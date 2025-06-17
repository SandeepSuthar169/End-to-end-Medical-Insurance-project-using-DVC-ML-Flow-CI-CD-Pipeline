import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os

def load_data(filepath: str) -> pd.DataFrame:
     try:
          return pd.read_csv(filepath)
     except Exception as e:
          raise Exception(f"error loading  data from {filepath}")
     
# train_data = pd.read_csv("./data/raw/train.csv")
# test_data = pd.read_csv("./data/raw/test.csv")

def transform_data(df):
     try:
          Pipeline  = ColumnTransformer(transformers=[
               ('onehot', OneHotEncoder(handle_unknown='ignore'),[1,5,4]),
               ('Scaler', StandardScaler(), [0, 2])
               ])
          transfor = Pipeline.fit_transform(df)
          transfor_df = pd.DataFrame(transfor)
          return transfor_df
     except Exception as e:
          raise Exception(f"error data transform values:{e}")
     


def save_data(df: pd.DataFrame, filepath) -> None:
     try:
          df.to_csv(filepath, index=False)
     except Exception as e:
          raise Exception(f"error saving data to {filepath}: {e}")
          


     # train_transform_data = transform_data(train_data)
     # test_transform_data = transform_data(test_data)


def main():
     try:
          raw_data_path = "./data/raw/"
          transform_data_path= "./data/processed"

          train_data = load_data(os.path.join(raw_data_path, "train.csv"))
          test_data = load_data(os.path.join(raw_data_path, "test.csv"))
          
          train_transform_data = transform_data(train_data)
          test_transform_data = transform_data(test_data)

          os.makedirs(transform_data_path)




     # data_path = os.path.join("data", "processed")
     # os.makedirs(data_path)

          save_data(train_transform_data,os.path.join(transform_data_path, "train_transform.csv"))
          save_data(test_transform_data,os.path.join(transform_data_path,"test_transform.csv"))
     except Exception as e:
          raise Exception(f"error occurred:{e}")
     

if __name__ == "__main__":
     main()
