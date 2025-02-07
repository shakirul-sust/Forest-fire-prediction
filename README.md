# Forest Fire Prediction

## Overview

This project intends to forecast the occurrence of forest fires using various machine learning models. The dataset is preprocessed, exploratory data analysis (EDA) is performed, and multiple models are trained to predict fire risk based on different features.

## Features

- **Data Preprocessing**: Cleaning and transforming categorical data into numerical data
- **Exploratory Data Analysis (EDA)**: Making charts and looking at missing and distribution values
- **Feature Engineering**: Removing not needed columns and changing the numerical features
- **Machine Learning Models**: Linear Regression, Decision Tree, and Random Forest
- **Model Evaluation**: RMSE calculation for model comparison
- **Model Saving**: Models are trained and saved via the script

## Dataset

The dataset file `forest_fire.csv` with all the features that cause the forest to burn down such as fire hot spots, railways, highways, etc. is the one used. To prepare the data the following transformations are made: the data preprocessing code takes out columns such as (`day`, `month`, and `year`). It encodes categorical variables instead of solving the problem manually.

## Installation

Some dependencies are required to be installed as follows:

```sh
pip install -r requirements.txt
```

## Running the Project

1. Copy the dataset to the `data/` directory
2. Run the main script:

```sh
python main.py
```

## Dependencies

For this project the following Python libraries are employed:

- `pandas`
- `numpy`
- `matplotlib`

The library dependencies, including seaborn, scikit-learn, and joblib, are stated in requirements.txt

## Model Training

- Data is separated into training and testing parts, which are then used for building and testing the model successively.
- Missing values are imputed using a `SimpleImputer`.
- The data is standardized using the `StandardScaler`.
- Three models are trained. These models are:
  - **Linear Regression**
  - **Decision Tree Regression**
  - **Random Forest Regression**
- The model evaluation takes into account the **Root Mean Squared Error (RMSE)**.

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

