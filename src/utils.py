import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.metrics import r2_score
import dill

import pickle


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates a list of machine learning models on given training and test data.

    Args:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        models (dict): Dictionary where keys are model names and values are model objects.

    Returns:
        dict: Dictionary containing the test scores (R2 score) for each model.
    """

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_object(file_path):
        
    try:
        with open(file_path,"rb") as file_obj:
           return dill.load(file_obj)
           #return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)    

