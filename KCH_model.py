import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import statsmodels.formula.api as smf


king_county_house = pd.read_csv("data/King_County_House_prices_dataset.csv", delimiter=",")

# Set correct format
king_county_house["price"] = king_county_house["price"].astype("int")
king_county_house["grade"] = king_county_house["grade"].astype("category")

# Remove outlier
king_county_house = king_county_house[king_county_house["bedrooms"] != 33]

# Feature selection
X = king_county_house[["grade", "sqft_living", "sqft_above", "sqft_living15", "bathrooms"]]
Y = king_county_house["price"]

# Splitting data
print("-----  Splitting the data in train and test ----")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Adding the constant
X_train = sm.add_constant(X_train) # adding a constant
X_test = sm.add_constant(X_test) # adding a constant

# Training the model
print("-----  Training the model ----")
model = sm.OLS(y_train, X_train).fit()
print_model = model.summary()

# Predictions to check the model
print("-----  Evaluating the model ----")
predictions = model.predict(X_train)
err_train = np.sqrt(mean_squared_error(y_train, predictions))
predictions_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, predictions_test))


print(print_model)
print ("-------------")
print (f"RMSE on train data: {err_train}")
print (f"RMSE on test data: {err_test}")
