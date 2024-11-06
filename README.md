# EX.NO.09        A project on Time series analysis on student score study hours using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on student score study hours using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of student score and study hours 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Step 1: Load Data
# Assuming data is in CSV format, replace 'data.csv' with the actual file path
# data = pd.read_csv('data.csv')
# For demonstration, manually creating the dataset here
# Creating the dataset manually (since it's provided in an image)
df = pd.read_csv('score.csv')

# Step 2: Plot the Time Series
plt.plot(data['Scores'])
plt.title("Scores over Time")
plt.xlabel("Index")
plt.ylabel("Scores")
plt.show()

# Step 3: ADF Test for Stationarity
result = adfuller(data['Scores'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] > 0.05:
    print("Series is non-stationary")
else:
    print("Series is stationary")

# Step 4: Differencing if necessary
data['Scores_diff'] = data['Scores'].diff().dropna()
plt.plot(data['Scores_diff'].dropna())
plt.title("Differenced Scores")
plt.show()

# Step 5: ACF and PACF plots
plot_acf(data['Scores'].dropna())
plt.show()
plot_pacf(data['Scores'].dropna())
plt.show()

# Step 6: ARIMA Model Selection (Auto-fit would require `pmdarima` package if you want to skip p, d, q selection)
# Here we'll set p=1, d=1 (after differencing), q=1 as an initial guess; adjust based on ACF and PACF

model = ARIMA(data['Scores'], order=(1, 1, 1))
arima_result = model.fit()
print(arima_result.summary())

# Step 7: Forecasting
forecast = arima_result.forecast(steps=5)
print("Forecasted Values:", forecast)

# Step 8: Model Evaluation with Mean Squared Error
train_size = int(len(data) * 0.8)
train, test = data['Scores'][:train_size], data['Scores'][train_size:]
model = ARIMA(train, order=(1, 1, 1))
fitted_model = model.fit()
predictions = fitted_model.forecast(len(test))
mse = mean_squared_error(test, predictions)
print("Mean Squared Error:", mse)

# Optionally, plot predictions vs actual values
plt.plot(test.values, label='Actual')
plt.plot(predictions, label='Predicted', color='red')
plt.legend()
plt.title("ARIMA Predictions vs Actual Scores")
plt.show()


### OUTPUT:
![download](https://github.com/user-attachments/assets/a2724231-f9b9-4c98-84c1-8ef7bf123f94)

![download](https://github.com/user-attachments/assets/0bf94de7-d44a-4264-8452-ec68916b902c)

![download](https://github.com/user-attachments/assets/458b8a13-eb10-4993-be77-f7bdf59871cf)

![download](https://github.com/user-attachments/assets/7970fa27-276b-414d-9b61-de37547a9057)

![download](https://github.com/user-attachments/assets/86240b74-af60-4fec-8420-9691e1d85b60)

![image](https://github.com/user-attachments/assets/217f403d-7e57-4b29-a11e-406717346756)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
