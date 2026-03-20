import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create a preprocessing pipeline for all numerical features.
        California Housing dataset from sklearn has no categorical features -
        all features are already numerical.
        """
        try:
            numerical_columns = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude',
                'rooms_per_household', 'bedrooms_ratio', 'population_per_household'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            # --- Feature Engineering ---
            for df in [train_df, test_df]:
                # Average rooms is already per household in sklearn dataset,
                # but let's create interaction features
                df['rooms_per_household'] = df['AveRooms'] * df['AveOccup']
                df['bedrooms_ratio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-6)
                df['population_per_household'] = df['Population'] / (df['AveOccup'] + 1e-6)

            logging.info("Feature engineering completed")

            # --- Get preprocessor ---
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "MedHouseVal"
            input_feature_columns = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude',
                'rooms_per_household', 'bedrooms_ratio', 'population_per_household'
            ]

            input_feature_train_df = train_df[input_feature_columns]
            target_feature_train = np.log1p(train_df[target_column].values)

            input_feature_test_df = test_df[input_feature_columns]
            target_feature_test = np.log1p(test_df[target_column].values)

            logging.info("Applying preprocessing (impute + scale)")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            train_arr = np.c_[input_feature_train_arr, target_feature_train]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Preprocessor saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
