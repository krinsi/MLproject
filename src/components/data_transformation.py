import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure the project root directory is added to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# Now import the 'src' module
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            # Specify the columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course",
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get preprocessor
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Specify the target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split into features and target variables
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply preprocessing on train and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into final train and test arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
