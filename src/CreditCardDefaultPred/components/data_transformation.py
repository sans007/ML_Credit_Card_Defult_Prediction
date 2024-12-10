import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import SMOTE

from src.CreditCardDefaultPred.utils import save_object

from src.CreditCardDefaultPred.exception import CustomException
from src.CreditCardDefaultPred.logger import logging
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
       
        """
        This function is responsible for data transformation.
        It balances the dataset using SMOTE, applies feature engineering, 
        and returns the transformed DataFrame and a list of feature column names.
        """
        try:
            # Initialize SMOTE
            smote = SMOTE(random_state=42)
            
            # Separate features and target
            X = self.iloc[:, :-1]
            y = self['default_payment_next_month']
            
            # Apply SMOTE
            x_smote, y_smote = smote.fit_resample(X, y)
            
            # Combine resampled data
            self_fr = pd.DataFrame(x_smote, columns=self.columns[:-1])
            self_fr['default'] = y_smote
            
            # Rename columns
            self_fr.rename(columns={
                'PAY_0': 'PAY_SEPT', 'PAY_2': 'PAY_AUG', 'PAY_3': 'PAY_JULY', 
                'PAY_4': 'PAY_JUNE', 'PAY_5': 'PAY_MAY', 'PAY_6': 'PAY_APRIL',
                'PAY_AMT1': 'PAY_AMT_SEPT', 'PAY_AMT2': 'PAY_AMT_AUG', 'PAY_AMT3': 'PAY_AMT_JULY',
                'PAY_AMT4': 'PAY_AMT_JUNE', 'PAY_AMT5': 'PAY_AMT_MAY', 'PAY_AMT6': 'PAY_AMT_APRIL',
                'BILL_AMT1': 'BILL_AMT_SEPT', 'BILL_AMT2': 'BILL_AMT_AUG', 'BILL_AMT3': 'BILL_AMT_JULY',
                'BILL_AMT4': 'BILL_AMT_JUNE', 'BILL_AMT5': 'BILL_AMT_MAY', 'BILL_AMT6': 'BILL_AMT_APRIL'
            }, inplace=True)
            
            # Modify categorical features
            self_fr['EDUCATION'] = np.where(self_fr['EDUCATION'].isin([0, 5, 6]), 4, self_fr['EDUCATION'])
            self_fr['MARRIAGE'] = np.where(self_fr['MARRIAGE'] == 0, 3, self_fr['MARRIAGE'])
            
            # Replace categorical values with labels
            self_fr.replace({
                'SEX': {1: 'MALE', 2: 'FEMALE'},
                'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'},
                'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}
            }, inplace=True)
            
            # Drop irrelevant columns
            self_fr.drop(columns=['ID'], inplace=True)
            
            # One-Hot Encoding
            self_fr = pd.get_dummies(self_fr, columns=['SEX', 'EDUCATION', 'MARRIAGE']).astype(int)
            self_fr = pd.get_dummies(self_fr, columns=['PAY_SEPT', 'PAY_AUG', 'PAY_JULY', 'PAY_JUNE', 'PAY_MAY', 'PAY_APRIL'], drop_first=True).astype(int)
            
            # Get feature column names
            self_fr=self_fr.drop(columns=['default'])
            use_col = self_fr.columns.tolist()
            
            col_pipeline = Pipeline(steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
            ])
            
            preprocessor=ColumnTransformer(
                    [
                        ("col_pipeline",col_pipeline,use_col),
                        
                    ]
                )
            
            return preprocessor



        except Exception as e:
            raise CustomException(e,sys)