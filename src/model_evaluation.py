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

def load_data(filepath: str) -> pd.DataFrame:
   try:
        return pd.read_csv(filepath)
   except Exception as e:
       raise Exception(f"error load data {filepath}: {e}")
   



# test_data = pd.read_csv("./data/processed/test_transform.csv")

# X_test = test_data.iloc[:, :-1].values
# y_test = test_data.iloc[:, -1].values


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.iloc[:, 0:-1].values
        y = data.iloc[:,-1].values
        return X,y
    except Exception as e:
        raise Exception(f"error  preparing data:{e}")
    


# Load the model

# with open("model.pkl", "rb") as file:
#     model = pickle.load(file)

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
            print(f"Loaded model type: {type(model)}")  # Debug print
            return model
    except Exception as e:
        raise Exception(f"Error loading model {filepath}: {e}")

     


# Predict
def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:    
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
        return metrics_dict
    except Exception as e:
        raise Exception(f"error of evaluatuion model {e}")
        
# Save metrics

def save_metrics(metrics_dict:dict, filepath:str) -> None:
    try:
        with open(filepath, 'w') as file:  # correct spelling
          json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"error saving metrics to {filepath} :{e}")
    
def main():
    try:
        test_data_path = "./data/processed/test_transform.csv"
        model_path = "model.pkl"
        metrics_path = "metrics.json"

        test_data=load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model =load_model(model_path)
        metrics  =evaluation_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:  
        raise Exception(f"error occurred:{e}") 
     

if __name__ == "__main__":
    main()