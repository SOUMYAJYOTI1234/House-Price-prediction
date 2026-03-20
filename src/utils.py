import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Save a Python object to a file using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object from a file using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Train and evaluate multiple models with hyperparameter tuning.
    Returns a dict of {model_name: R² score on test set}.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")

            param = params.get(model_name, {})

            if param:
                gs = GridSearchCV(
                    model, param, cv=3, scoring='r2', n_jobs=-1, verbose=0
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                logging.info(f"{model_name} best params: {gs.best_params_}")

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)

            report[model_name] = r2
            logging.info(
                f"{model_name} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
