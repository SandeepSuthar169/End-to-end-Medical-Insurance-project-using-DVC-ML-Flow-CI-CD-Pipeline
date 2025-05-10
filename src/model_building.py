import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.ensemble import RandomForestRegressor

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        n_estimators  = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        min_samples_split = params['model_building']['min_samples_split']
        return n_estimators, max_depth, min_samples_split
    
    except Exception as e:
        raise Exception(f"error load params from {params_path}:{e}")

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error loading data form {filepath}: {e}")
    
# train_data = pd.read_csv("./data/processed/train_transform.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.iloc[:, 0:-1].values
        y = data.iloc[:,-1].values
        return X,y
    except Exception as e:
        raise Exception(f"error  preparing data:{e}")
    

    
# X_train = train_data.iloc[:,0:-1].values
# y_train = train_data.iloc[:,-1].values


# rfr =RandomForestRegressor(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_samples_split)
# rfr.fit(X_train, y_train)
def train_model(X:pd.DataFrame, y:pd.Series, n_estimators: int, max_depth: int, min_samples_split: int) -> RandomForestRegressor:
    try:
        rfr =RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split)
        rfr.fit(X, y)
        return rfr
    except Exception as e:
        raise Exception(f"error training model: {e}")
        

# pickle.dump(rfr, open("model.pkl", "wb"))

def save_model(model: RandomForestRegressor, filepath) -> None:
    try:
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"error saveing model to  {filepath}: {e}")

def main():
    try:
        params_path = 'params.yaml'
        data_path = './data/processed/train_transform.csv'
        model_name = 'model.pkl'

        n_estimators, max_depth, min_samples_split = load_params(params_path)

        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators, max_depth, min_samples_split)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"an error occurred : {e}") 
    
if __name__ =="__main__":
    main()
       

