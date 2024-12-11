from src.CreditCardDefaultPred.logger import logging
from src.CreditCardDefaultPred.exception import CustomException
from src.CreditCardDefaultPred.components.data_ingestion import DataIngestion
from src.CreditCardDefaultPred.components.data_ingestion import DataIngestionConfig
from src.CreditCardDefaultPred.components.data_transformation import DataTransformation, DataTransformationConfig
import sys


if __name__ == '__main__':
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)