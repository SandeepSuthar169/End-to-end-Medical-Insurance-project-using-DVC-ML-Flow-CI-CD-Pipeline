import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os

train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")

def transform_data(df):
     Pipeline  = ColumnTransformer(transformers=[
          ('onehot', OneHotEncoder(handle_unknown='ignore'),[1,5,4]),
          ('Scaler', StandardScaler(), [0, 2])
          ])
     transfor = Pipeline.fit_transform(df)
     transfor_df = pd.DataFrame(transfor)
     return transfor_df

train_transform_data = transform_data(train_data)
test_transform_data = transform_data(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path)

train_transform_data.to_csv(os.path.join(data_path, "train_transform.csv"),index = False)
test_transform_data.to_csv(os.path.join(data_path,"test_transform.csv"), index=False)

