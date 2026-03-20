import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(
                    objective='reg:squarederror', random_state=42, verbosity=0
                ),
            }

            params = {
                "Linear Regression": {},
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0],
                },
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                },
                "XGBoost": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7],
                },
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params,
            )

            # Get the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No model achieved acceptable R² score (>0.6)", sys
                )

            logging.info(
                f"Best model: {best_model_name} with R²={best_model_score:.4f}"
            )

            # Print all model scores
            print("\n" + "=" * 60)
            print("MODEL COMPARISON RESULTS")
            print("=" * 60)
            for name, score in sorted(
                model_report.items(), key=lambda x: x[1], reverse=True
            ):
                marker = " ⭐ BEST" if name == best_model_name else ""
                print(f"  {name:25s}  R² = {score:.4f}{marker}")
            print("=" * 60)
            print(f"\n✅ Best Model: {best_model_name} (R² = {best_model_score:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info("Best model saved")

            # Final prediction for verification
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
