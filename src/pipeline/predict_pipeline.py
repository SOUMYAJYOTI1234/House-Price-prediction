import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            # Reverse the log1p transformation to get actual prices
            actual_prices = np.expm1(preds)

            return actual_prices

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Maps form inputs to a DataFrame for prediction."""

    def __init__(
        self,
        MedInc: float,
        HouseAge: float,
        AveRooms: float,
        AveBedrms: float,
        Population: float,
        AveOccup: float,
        Latitude: float,
        Longitude: float,
    ):
        self.MedInc = MedInc
        self.HouseAge = HouseAge
        self.AveRooms = AveRooms
        self.AveBedrms = AveBedrms
        self.Population = Population
        self.AveOccup = AveOccup
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "MedInc": [self.MedInc],
                "HouseAge": [self.HouseAge],
                "AveRooms": [self.AveRooms],
                "AveBedrms": [self.AveBedrms],
                "Population": [self.Population],
                "AveOccup": [self.AveOccup],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
                "rooms_per_household": [self.AveRooms * self.AveOccup],
                "bedrooms_ratio": [self.AveBedrms / (self.AveRooms + 1e-6)],
                "population_per_household": [
                    self.Population / (self.AveOccup + 1e-6)
                ],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
