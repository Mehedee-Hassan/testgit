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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For illustration, generate synthetic data
np.random.seed(0)  # For reproducibility
dates = pd.date_range('2023-01-01', periods=100)
actual = np.random.rand(100) * 100
forecast1 = actual + (np.random.rand(100) - 0.5) * 10
forecast2 = actual + (np.random.rand(100) - 0.5) * 15

# Create a DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Actual': actual,
    'Forecast1': forecast1,
    'Forecast2': forecast2
})

# Calculate errors
df['AE1'] = np.abs(df['Actual'] - df['Forecast1'])
df['AE2'] = np.abs(df['Actual'] - df['Forecast2'])
df['SE1'] = np.square(df['Actual'] - df['Forecast1'])
df['SE2'] = np.square(df['Actual'] - df['Forecast2'])

# Plotting
plt.figure(figsize=(14, 8))

# Absolute Error plot
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['AE1'], label='Absolute Error Forecast 1')
plt.plot(df['Date'], df['AE2'], label='Absolute Error Forecast 2')
plt.title('Absolute Error Over Time')
plt.legend()

# Squared Error plot
plt.subplot(2, 1, 2)
plt.plot(df['Date'], df['SE1'], label='Squared Error Forecast 1')
plt.plot(df['Date'], df['SE2'], label='Squared Error Forecast 2')
plt.title('Squared Error Over Time')
plt.legend()

plt.tight_layout()
plt.show()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame creation - replace this with your actual DataFrame loading or creation
np.random.seed(0)  # For reproducibility
dates = pd.date_range('2023-01-01', periods=100)
actual_values = np.random.rand(100) * 100
forecast_values = actual_values + (np.random.rand(100) - 0.5) * 10
df = pd.DataFrame({'Date': dates, 'Actual': actual_values, 'Forecast': forecast_values})

# Optionally copy Actual to create a simulated forecast for comparison
# For illustration, we skip this since we already have a forecast

# Calculate Absolute Error (AE) and Squared Error (SE)
df['AE'] = np.abs(df['Actual'] - df['Forecast'])
df['SE'] = np.square(df['Actual'] - df['Forecast'])

# Plotting
plt.figure(figsize=(14, 8))

# Actual vs. Forecast plot
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['Actual'], label='Actual Values', marker='o', linestyle='-', markersize=5)
plt.plot(df['Date'], df['Forecast'], label='Forecast Values', marker='x', linestyle='--', markersize=5)
plt.title('Actual vs. Forecast Values')
plt.legend()

# Error plot (Optional) - Absolute Error in this case
plt.subplot(2, 1, 2)
plt.plot(df['Date'], df['AE'], label='Absolute Error', color='red', linestyle='-', linewidth=2)
plt.title('Absolute Error Over Time')
plt.legend()

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


shapiro_test = stats.shapiro(residuals)
print(f'Shapiro-Wilk test statistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}')






##################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generating a sample dataset
np.random.seed(0)
num_points = 10000
actual = np.random.normal(100, 15, num_points)
predictions = actual + np.random.normal(0, 10, num_points)  # Predictions with some error

df = pd.DataFrame({'Actual': actual, 'Prediction': predictions})

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Actual'], df['Prediction'], alpha=0.3)  # Adjust alpha for transparency
plt.title('Actual vs. Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()





plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Actual'], label='Actual Values', alpha=0.6)
plt.plot(df.index, df['Prediction'], label='Predicted Values', alpha=0.6)
plt.legend()
plt.title('Actual and Predicted Values Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(df['Actual'], label='Actual Values', bw_adjust=0.5)
sns.kdeplot(df['Prediction'], label='Predicted Values', bw_adjust=0.5)
plt.legend()
plt.title('Density Plot of Actual and Predicted Values')
plt.show()



plt.figure(figsize=(10, 6))
plt.hexbin(df['Actual'], df['Prediction'], gridsize=50, cmap='Purples', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Hexbin Plot of Actual vs. Predicted Values')
plt.show()

