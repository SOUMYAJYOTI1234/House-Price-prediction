import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException


def run_training_pipeline():
    """Run the complete training pipeline."""
    try:
        print("🏠 House Price Prediction - Training Pipeline")
        print("=" * 50)

        # Step 1: Data Ingestion
        print("\n📥 Step 1: Data Ingestion...")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        print(f"   ✅ Data saved to artifacts/")

        # Step 2: Data Transformation
        print("\n🔄 Step 2: Data Transformation & Feature Engineering...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(train_path, test_path)
        )
        print(f"   ✅ Preprocessor saved to {preprocessor_path}")

        # Step 3: Model Training
        print("\n🤖 Step 3: Model Training & Selection...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"\n🎉 Training Complete! Final R² Score: {r2_score:.4f}")
        return r2_score

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
