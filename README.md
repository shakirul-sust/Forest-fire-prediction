# Forest Fire Prediction

## Overview

This project aims to predict forest fire occurrences using various machine learning models. The dataset is preprocessed, exploratory data analysis (EDA) is performed, and multiple models are trained to predict fire risk based on different features.

## Features

- **Data Preprocessing**: Cleaning and encoding categorical data
- **Exploratory Data Analysis (EDA)**: Visualizing missing values and distributions
- **Feature Engineering**: Dropping unnecessary columns and transforming numerical features
- **Machine Learning Models**: Linear Regression, Decision Tree, and Random Forest
- **Model Evaluation**: RMSE calculation to compare model performance
- **Model Saving**: Trained models are saved for later use

## Dataset

The dataset used is `forest_fire.csv`, which contains various features related to forest fire occurrences. The preprocessing script removes unnecessary columns (`day`, `month`, `year`) and encodes categorical variables.

## Installation

To run this project, install the required dependencies using:

```sh
pip install -r requirements.txt
```

## Running the Project

1. Place the dataset inside the `data/` directory.
2. Run the main script:

```sh
python main.py
```

## Dependencies

This project uses several Python libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

All dependencies are listed in `requirements.txt`.

## Model Training

- The dataset is split into training and testing sets.
- Missing values are handled using `SimpleImputer`.
- Data is scaled using `StandardScaler`.
- Three models are trained:
  - **Linear Regression**
  - **Decision Tree Regression**
  - **Random Forest Regression**
- Models are evaluated based on **Root Mean Squared Error (RMSE)**.

## Saving Models

The trained models and preprocessing tools are saved in the `models/` directory:

```sh
models/random_forest.pkl
models/num_imputer.pkl
models/num_scaler.pkl
```

## License

This project is licensed under the MIT License.

## Author

Developed by Shakirul

