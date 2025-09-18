import os
import sys

import numpy as np 
import pandas as pd
import pickle

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for model_name, model in models.items():

            # get parameter grid for this model
            para = param.get(model_name, {})

            if para:  # if hyperparameters provided
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring="r2")
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # evaluate
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score  

        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)