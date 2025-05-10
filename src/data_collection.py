import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



df = pd.read_csv(r"C:\Users\Sande\Desktop\Datasets\medical_insurance.csv")

train_data, test_data = train_test_split(df, random_state=42, test_size=0.2)

data_path = os.path.join("data", "raw")

os.makedirs(data_path, exist_ok=True)
train_data.to_csv(os.path.join(data_path,"train_data"), index= False)
test_data.to_csv(os.path.join(data_path, "test_data"), index= False)

