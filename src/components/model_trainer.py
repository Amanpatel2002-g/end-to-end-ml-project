import os, sys
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaulate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge":Ridge(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor()
            }
            model_report:dict = evaulate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models)
            
            ## To get the best model sccore from dict
            best_model_score = max(sorted(list(model_report.values())))
            
            ## To get th best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            
            logging.info("best model found on both training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                        )
            
            predicted = best_model.predict(X_test)
            
            r2 = r2_score(y_test, predicted)
            
            return r2
        except Exception as e:
            raise CustomException(e, sys)