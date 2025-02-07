import joblib
import numpy as np

forest_reg = joblib.load('models/random_forest.pkl')
num_imputer = joblib.load('models/num_imputer.pkl')
num_scaler = joblib.load('models/num_scaler.pkl')

# load the test data
import pandas as pd

test = pd.read_csv('data/test.csv')

# separate features and target
X_test = test.drop('FWI', axis=1)
y_test = test['FWI'].copy()

# separate numerical and categorical features
num_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_test.select_dtypes('object').columns.tolist()

X_test_num = X_test[num_features]
X_test_cat = X_test[cat_features]

print(f'Numerical features: {num_features}')
print(f'Categorical features: {cat_features}')

# apply transformations
X_test_num = num_imputer.transform(X_test_num)
X_test_num = num_scaler.transform(X_test_num)


# make predictions
y_pred = forest_reg.predict(X_test_num)

# evaluate the model
from sklearn.metrics import root_mean_squared_error

rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))

print(f'RMSE: {rmse}')