import os
import sys
# Adding project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")
        try:
            # Ensure the correct path to the dataset
            data_path = os.path.join('notebook', 'data', 'diabetes.csv')
            df = pd.read_csv(data_path)
            logging.info("Dataset loaded successfully as a DataFrame.")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully.")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test datasets created and saved successfully.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {fnf_error}")
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)


# Placeholder for DataTransformation class
class DataTransformation:
    def initiate_data_transformation(self, train_data, test_data):
        logging.info("Data transformation started.")
        # Implement the transformation logic here
        logging.info("Data transformation completed.")


if __name__ == "__main__":
        # Perform data ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Perform data transformation
        
        data_transformation = DataTransformation()
       ## train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        train_arr, test_arr, train_labels, test_labels = data_transformation.initiate_data_transformation(train_data, test_data)

        # Perform model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        
        
       