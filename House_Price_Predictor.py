import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Training Data:")
print(train_data.head())
print("Testing Data:")
print(test_data.head())

print("Missing values in training data:\n", train_data.isnull().sum())
print("Missing values in testing data:\n", test_data.isnull().sum())

test_data["SalePrice"] = np.nan
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

combined_data = pd.get_dummies(combined_data, drop_first=True)

train_processed = combined_data[~combined_data["SalePrice"].isna()]
test_processed = combined_data[combined_data["SalePrice"].isna()]

test_processed = test_processed.drop("SalePrice", axis=1)

X_train = train_processed.drop(["SalePrice"], axis=1)
Y_train = train_processed["SalePrice"]

X_test = test_processed

model = XGBRegressor()
model.fit(X_train, Y_train)

training_data_prediction = model.predict(X_train)

r2_score_train = metrics.r2_score(Y_train, training_data_prediction)
mae_train = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("Training R-squared: ", r2_score_train)
print("Training Mean Absolute Error: ", mae_train)

test_predictions = model.predict(X_test)

print("Test Predictions:")
print(test_predictions)

output = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_predictions
})
output.to_csv("house_price_predictions.csv", index=False)
print("Predictions saved to 'house_price_predictions.csv'")

plt.figure(figsize=(8, 6))
plt.scatter(Y_train, training_data_prediction, alpha=0.5, color='blue')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Prices (Training Data)")
plt.show()



