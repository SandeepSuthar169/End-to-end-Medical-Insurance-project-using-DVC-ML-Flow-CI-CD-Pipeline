import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.ensemble import RandomForestRegressor


with open("params.yaml") as file:
    params = yaml.safe_load(file)

n_estimators  = params['model_building']['n_estimators']
max_depth = params['model_building']['max_depth']
min_samples_split = params['model_building']['min_samples_split']

train_data = pd.read_csv("./data/processed/train_transform.csv")

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

rfr =RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split)
rfr.fit(X_train, y_train)

pickle.dump(rfr, open("model.pkl", "wb"))