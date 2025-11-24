import os
import sys

import numpy as np 
import pandas as pd
import dill

from exception.exception import CustomException
from logger.logger import get_logger
from sklearn.metrics import r2_score

def save_object(file_path,object):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_object:
            dill.dump(object,file_object)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,x_test,y_train,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            
            # Fit the model first
            model.fit(x_train, y_train)

            y_train_predict=model.predict(x_train)
            y_test_predict=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_predict)
            test_model_score=r2_score(y_test,y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report 
        
    except Exception as e:
        raise CustomException(e,sys)
