import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error,
    mean_squared_error
)

test_data = pd.read_csv("./data/processed/test_transform.csv")

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

metrics_dict = {
    'r2': r2,
    'mae': mae,
    'mse': mse
}

# Save metrics
with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
