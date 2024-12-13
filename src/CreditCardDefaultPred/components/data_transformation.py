import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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


    # def get_data_transformer_object(self,df1,df2):
       
    #     """
    #     This function is responsible for data transformation.
    #     It balances the dataset using SMOTE, applies feature engineering, 
    #     and returns the transformed DataFrame and a list of feature column names.
    #     """
    #     try:
    #         # Initialize SMOTE
    #         smote = SMOTE(random_state=42)

    #         # Separate features and target
    #         X = df.iloc[:, :-1]
    #         y = df['default_payment_next_month']

    #         # Apply SMOTE
    #         x_smote, y_smote = smote.fit_resample(X, y)

    #         # Combine resampled data
    #         df_fr = pd.DataFrame(x_smote, columns=df.columns[:-1])
    #         df_fr['default_payment_next_month'] = y_smote


    #         # Modify categorical features
    #         df_fr['EDUCATION'] = np.where(df_fr['EDUCATION'].isin([0, 5, 6]), 4, df_fr['EDUCATION'])
    #         df_fr['MARRIAGE'] = np.where(df_fr['MARRIAGE'] == 0, 3, df_fr['MARRIAGE'])

    #         # Replace categorical values with labels
    #         df_fr.replace({
    #             'SEX': {1: 'MALE', 2: 'FEMALE'},
    #             'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'},
    #             'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}
    #         }, inplace=True)

    #         # Drop irrelevant columns
    #         df_fr.drop(columns=['ID'], inplace=True)


    #         # Get feature column names
    #         df_fr=df_fr.drop(columns=['default_payment_next_month'])
    #         enco_columns=['SEX', 'EDUCATION', 'MARRIAGE','PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    #         unenco_columns = [col for col in df_fr.columns if col not in enco_columns]


    #         unenco_columns_pipeline = Pipeline(steps=[
    #                 ("imputer",SimpleImputer(strategy='median')),
    #                 ('scalar',StandardScaler())
    #         ])

    #         enco_columns_pipeline = Pipeline(steps=[
    #                             ("imputer",SimpleImputer(strategy="most_frequent")),
    #                             ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
    #                             ("scaler",StandardScaler(with_mean=False))
    #         ])


    #         preprocessor=ColumnTransformer(
    #                 [
    #                     ("unenco_columns_pipeline",unenco_columns_pipeline,unenco_columns),
    #                     ("enco_columns_pipeline",enco_columns_pipeline,enco_columns)

    #                 ]
    #             )
            
    #         return preprocessor


    #     except Exception as e:
    #         raise CustomException(e,sys)
        

    # def initiate_data_transformation(self,train_path,test_path):
    #     try:
    #         train_df=pd.read_csv(train_path)
    #         test_df=pd.read_csv(test_path)

    #         # Ensure the files are loaded as DataFrames
    #         if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
    #           raise ValueError("Train and Test datasets must be pandas DataFrames.")

    #         logging.info("Reading the train and test file")

    #         preprocessing_obj = self.get_data_transformer_object(train_df, test_df)

    #         target_column_name = "default_payment_next_month"
            

    #         ## divide the train dataset to independent and dependent feature
    #         input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
    #         target_feature_train_df=train_df[target_column_name]
        
    #         ## divide the test dataset to independent and dependent feature
    #         input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
    #         target_feature_test_df=test_df[target_column_name]
            
    #         logging.info("Applying Preprocessing on training and test dataframe")

    #         input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

    #         train_arr = np.c_[
    #             input_feature_train_arr, np.array(target_feature_train_df)
    #         ]
    #         test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

    #         logging.info(f"Saved preprocessing object")

    #         save_object(
    #             file_path=self.data_transformation_config.preprocessor_obj_file_path,
    #             obj=preprocessing_obj
    #         )

    #         return (
    #             train_arr,
    #             test_arr,
    #             self.data_transformation_config.preprocessor_obj_file_path
    #         )

    def get_data_transformer_object(self, df):
        """
        This function constructs a preprocessing object with pipelines for both
        categorical and numerical data transformations.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            preprocessor (ColumnTransformer): The preprocessing pipeline object.
        """
        try:
            # Identify categorical and numerical columns
            enco_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            unenco_columns = [col for col in df.columns if col not in enco_columns and col != 'default_payment_next_month']

            # Pipelines for numerical and categorical columns
            unenco_columns_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            enco_columns_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Create ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("unenco_columns_pipeline", unenco_columns_pipeline, unenco_columns),
                    ("enco_columns_pipeline", enco_columns_pipeline, enco_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise Exception(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads, preprocesses, and balances datasets, then applies transformations.

        Args:
            train_path (str): Path to training data.
            test_path (str): Path to test data.

        Returns:
            tuple: Preprocessed training and test arrays and preprocessor object path.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test datasets.")

            # Apply SMOTE only on the training data
            smote = SMOTE(random_state=42)
            X_train = train_df.drop(columns=["default_payment_next_month", "ID"], errors="ignore")
            y_train = train_df["default_payment_next_month"]
            X_test = test_df.drop(columns=["default_payment_next_month", "ID"], errors="ignore")
            y_test = test_df["default_payment_next_month"]

            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test)


            # Feature engineering (same for train and test)
            for df in [X_train_smote, X_test_smote]:
                df['EDUCATION'] = np.where(df['EDUCATION'].isin([0, 5, 6]), 4, df['EDUCATION'])
                df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])

                df.replace({
                    'SEX': {1: 'MALE', 2: 'FEMALE'},
                    'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'},
                    'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}
                }, inplace=True)

                df.drop(columns=['ID'], inplace=True, errors="ignore")

            # Get preprocessor
            preprocessor_obj = self.get_data_transformer_object(X_train_smote)

            # Apply preprocessing
            X_train_arr = preprocessor_obj.fit_transform(X_train_smote)
            # X_test = test_df.drop(columns=["default_payment_next_month", "ID"], errors="ignore")
            # y_test = test_df["default_payment_next_month"]
            X_test_arr = preprocessor_obj.transform(X_test_smote)

            # Combine features and target
            train_arr = np.c_[X_train_arr, np.array(y_train_smote)]
            test_arr = np.c_[X_test_arr, np.array(y_test_smote)]

            logging.info("Saved preprocessing object.")

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys,e)