import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the CSV files
df1 = pd.read_csv('forecast1.csv')
df2 = pd.read_csv('forecast2.csv')

# Assuming the actual values are aligned and identical in both DataFrames
# and the index aligns them properly. If not, you might need to merge or align them first.

# Calculate error metrics for forecast1
mae1 = mean_absolute_error(df1['actual'], df1['forecast'])
mse1 = mean_squared_error(df1['actual'], df1['forecast'])
rmse1 = np.sqrt(mse1)
mape1 = np.mean(np.abs((df1['actual'] - df1['forecast']) / df1['actual'])) * 100

# Calculate error metrics for forecast2
mae2 = mean_absolute_error(df2['actual'], df2['forecast'])
mse2 = mean_squared_error(df2['actual'], df2['forecast'])
rmse2 = np.sqrt(mse2)
mape2 = np.mean(np.abs((df2['actual'] - df2['forecast']) / df2['actual'])) * 100

# Print the comparison
print(f"Forecast 1 - MAE: {mae1}, MSE: {mse1}, RMSE: {rmse1}, MAPE: {mape1}%")
print(f"Forecast 2 - MAE: {mae2}, MSE: {mse2}, RMSE: {rmse2}, MAPE: {mape2}%")

# Decide which forecast is better based on lower error metrics
