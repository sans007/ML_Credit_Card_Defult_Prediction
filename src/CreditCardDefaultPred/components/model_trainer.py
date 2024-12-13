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
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.CreditCardDefaultPred.logger import logging
from src.CreditCardDefaultPred.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    