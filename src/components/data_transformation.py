import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Dynamically add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Debug: Print the current system path
print("System Path:", sys.path)

# Import project-specific modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_objects(self):
        """
        Creates and returns the data transformation pipeline.
        """
        try:
            # Define numerical columns
            numerical_columns = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]

            # Define numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns preprocessing pipeline created.")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies data transformation to the train and test datasets.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded successfully.")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_objects()

            # Define target column
            target_column_name = "Outcome"

            # Separate features and target in train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Preprocessing object applied to training and testing datasets.")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

'''
if __name__ == "__main__":
    try:
        # Import `DataIngestion` after appending the path
        from src.components.data_ingestion import DataIngestion

        # Perform data ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Perform data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        logging.info("Data transformation completed successfully.")
        logging.info(f"Preprocessor saved at: {preprocessor_path}")

    except CustomException as e:
        logging.error(f"CustomException occurred: {e}")
    except Exception as ex:
        logging.error(f"Unhandled Exception: {ex}")
'''

