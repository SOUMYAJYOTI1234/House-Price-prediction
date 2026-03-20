# 🏠 California House Price Prediction

An end-to-end Machine Learning project to predict California house prices using advanced regression models.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-green?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?style=flat-square)

## 📌 Overview

This project predicts **median house prices** for California districts using census data. It compares **6 regression models**, auto-selects the best one, and serves predictions through a sleek Flask web application.

### Models Used
| Model | Type |
|-------|------|
| Linear Regression | Baseline |
| Ridge Regression | L2 Regularization |
| Lasso Regression | L1 Regularization |
| Random Forest | Ensemble |
| Gradient Boosting | Boosting |
| XGBoost | Advanced Boosting |

## 🏗️ Project Structure

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Fetch & split data
│   │   ├── data_transformation.py  # Feature engineering & scaling
│   │   └── model_trainer.py        # Train, tune & select best model
│   ├── pipeline/
│   │   ├── train_pipeline.py       # End-to-end training
│   │   └── predict_pipeline.py     # Single prediction interface
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form
├── artifacts/                      # Generated models & data
├── application.py                  # Flask web app
├── Dockerfile
├── requirements.txt
└── setup.py
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SOUMYAJYOTI1234/House-Price-prediction.git
cd House-Price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python -m src.pipeline.train_pipeline
```

This will:
- Download the California Housing dataset
- Engineer features (rooms per household, bedroom ratio, etc.)
- Train 6 models with hyperparameter tuning
- Auto-select and save the best model

### 4. Run the web app
```bash
python application.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

## 🐳 Docker

```bash
docker build -t house-price-predictor .
docker run -p 5000:5000 house-price-predictor
```

## 📊 Features

- **20,640+ data points** from the California Housing Census
- **Feature Engineering**: rooms per household, bedroom ratio, population per household
- **Hyperparameter Tuning** with GridSearchCV and 3-fold cross-validation
- **Log-transformed target** for better prediction distribution
- **Premium Flask UI** with dark glassmorphism theme
- **R² > 0.80** on test set

## 📝 Dataset

The project uses `sklearn.datasets.fetch_california_housing` — **no external CSV file needed**.

**Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

**Target**: MedHouseVal (median house value in $100,000s)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
