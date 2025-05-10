import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("./data/processed/train_transform.csv")

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

rfr =RandomForestRegressor()
rfr.fit(X_train, y_train)

pickle.dump(rfr, open("model.pkl", "wb"))