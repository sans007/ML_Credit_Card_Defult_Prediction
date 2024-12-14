from src.CreditCardDefaultPred.logger import logging
from src.CreditCardDefaultPred.exception import CustomException
from src.CreditCardDefaultPred.components.data_ingestion import DataIngestion
from src.CreditCardDefaultPred.components.data_ingestion import DataIngestionConfig
from src.CreditCardDefaultPred.components.data_transformation import DataTransformation, DataTransformationConfig
import sys
from src.CreditCardDefaultPred.components.model_trainer import ModelTrainer, ModelTrainerConfig


if __name__ == '__main__':
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        #model trainer
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)