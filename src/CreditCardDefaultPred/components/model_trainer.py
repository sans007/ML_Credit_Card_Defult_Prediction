import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.CreditCardDefaultPred.logger import logging
from src.CreditCardDefaultPred.exception import CustomException
from src.CreditCardDefaultPred.utils import evaluate_models,save_object,RandomizedSearchCV

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import dagshub



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        Accuracy_score = accuracy_score(actual, pred)
        Precision_score = precision_score(actual, pred)
        Recall_score = recall_score(actual, pred)
        F1_score = f1_score(actual, pred)
        return Accuracy_score, Precision_score, Recall_score, F1_score

    def initiate_model_trainer(self,train_array,test_array):
    
       try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        
            # Define models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Catboost Classifier": CatBoostClassifier(verbose=False),
            }

            # Define hyperparameters for RandomizedSearchCV
            params = {
                "Random Forest": {
                    "max_depth": [5, 8, 15, None],
                    "max_features": [5, 7, "sqrt", 8],
                    "min_samples_split": [2, 8, 15, 20],
                    "n_estimators": [100, 200, 500, 1000]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30]
                },
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [8, 16, 32, 64, 128],
                    "max_depth": [3, 5, 7],
                },
                "AdaBoost Classifier": {
                    "n_estimators": [8, 16, 32, 64, 128],
                    "learning_rate": [0.1, 0.01, 0.05]
                },
                "Catboost Classifier": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }

            }

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get the best model
            best_model_name = max(model_report, key=model_report.get)  # Get the key with the highest value
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"\nBest Model: {best_model_name} with Test Accuracy: {best_model_score:.4f}")

            # Fetch best parameters for the selected model
            best_param = params.get(best_model_name, "No hyperparameters available for this model.")
            print(f"Best Parameters for {best_model_name}: {best_param}")

            
            
            best_model.fit(X_train, y_train)

            
            mlflow
            dagshub.init(repo_owner='sans007', repo_name='ML_Credit_Card_Defult_Prediction', mlflow=True)
            mlflow.set_registry_uri("https://dagshub.com/sans007/ML_Credit_Card_Defult_Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                # Evaluate the model's performance
                Accuracy_score,Precision_score,Recall_score,F1_score = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_param)

                mlflow.log_metric("Accuracy", Accuracy_score)
                mlflow.log_metric("Precision", Precision_score)
                mlflow.log_metric("Recall", Recall_score)
                mlflow.log_metric("F1", F1_score)
                


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        

       
       except Exception as e:
           raise CustomException(e,sys)