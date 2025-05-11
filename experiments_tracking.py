import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error, 
    r2_score,
    
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

import dvclive
from dvclive import Live 
import yaml

n_estimators=yaml.safe_load(open("C:\Users\Sande\Desktop\project\medical\params.yaml"))["n_estimators"]

df = pd.read_csv("C:/Users/Sande/Desktop/Datasets/medical_insurance.csv")

df=df.drop_duplicates()

train_data, test_data = train_test_split(df, random_state=42, test_size=0.2) 

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:,-1].values

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:,-1].values  


Pipeline  = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'),[1,5,4]),
    ('Scaler', StandardScaler(), [0, 2])
])

# n_estimators = 1000
pipe = make_pipeline(Pipeline, RandomForestRegressor(n_estimators= n_estimators))
pipe.fit(X_train, y_train) 

y_pred = pipe.predict(X_test)
acc = r2_score(y_test, y_pred)
print(acc)




#-------------------------------- dvclive -------------------------------- 

with Live(save_dvc_exp = True) as live:    
    live.log_metric(acc*100)
    live.log_params("n_estimators", n_estimators)