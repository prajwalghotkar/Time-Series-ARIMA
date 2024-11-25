# Time-Series-ARIMA

# The ARIMA (AutoRegressive Integrated Moving Average) model is a widely used statistical method for time series forecasting. It combines three key components:
* AR (Autoregressive): This part captures the relationship between an observation and a number of lagged observations (previous time points).
* I (Integrated): This involves differencing the data to achieve stationarity, which means removing trends or seasonal structures that could affect the model's accuracy.
* MA (Moving Average): This component models the relationship between an observation and a residual error from a moving average model based on lagged observations.

# The ARIMA model is typically denoted as ARIMA(p, d, q), where:

p = number of lag observations included in the model (autoregressive part).
d = degree of differencing required to make the series stationary.
q = size of the moving average window.

# Steps to Implement ARIMA
1) Data Preparation:
  * Collect time series data, ensuring it is in chronological order.
  * Visualize the data to identify trends and seasonal patterns.
2) Check for Stationarity:
  * Use statistical tests like the Augmented Dickey-Fuller test to check if the series is stationary.
  * If not stationary, apply differencing until stationarity is achieved.
3) Identify Parameters (p, d, q):
  * Use Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to determine appropriate values for 
4) Fit the ARIMA Model:
  * Use statistical software or libraries (e.g., statsmodels in Python) to fit the ARIMA model to your data.
5) Model Evaluation:
  * Evaluate the model using metrics such as R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage 
    Error (MAPE).
6) Forecasting:
  * Once the model is validated, use it to forecast future values in your time series.

# Example: Milk Production Forecasting Project
  A practical application of ARIMA can be seen in a project focused on forecasting milk production using historical data from January 1962 to December 1975. The project 
  involved:
* Analyzing monthly milk production data, which was identified as non-stationary.
* Applying the ARIMA model to forecast future production levels effectively.
* Evaluating the model's performance with various metrics, confirming its accuracy in predictions.
This project illustrates how ARIMA can be utilized in real-world scenarios, particularly in industries like dairy farming where accurate forecasting is crucial for operational planning and decision-making.

# This guide outlines how to apply the ARIMA (AutoRegressive Integrated Moving Average) model to three datasets: BOE-XUDLERD.csv, monthly-milk-production.csv, and 
  Time_Series_Temp_Data.csv. Each dataset represents a time series that can be analyzed for trends and forecasting future values.
# Steps for Implementing ARIMA
1) Data Preparation:
   Load each dataset and ensure they are in the correct time series format, with a datetime index.
2) Stationarity Check:
   Use the Augmented Dickey-Fuller (ADF) test to assess whether the time series is stationary. If the p-value is greater than 0.05, the series is non-stationary and will 
   require differencing.
3) Differencing:
   Apply differencing to stabilize the mean of the time series by removing trends or seasonality.
4) Parameter Identification (p, d, q):
   Analyze ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots to determine suitable values for p and q.
5) Model Fitting:
   Fit the ARIMA model using the identified parameters for each dataset.
6) Model Evaluation:
   Evaluate the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
7) Forecasting:
   Use the fitted model to forecast future values based on historical data.
# Example Implementation in Python
Below is a Python code snippet demonstrating how to apply the ARIMA model to each of the three datasets:
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to apply ARIMA model
def apply_arima(file_path):
    # Load dataset
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    # Check for stationarity
    result = adfuller(data['value'])
    print(f'ADF Statistic: {result[0]}, p-value: {result[1]}')

    # Differencing if necessary
    if result[1] > 0.05:
        data['value'] = data['value'].diff().dropna()

    # Plot ACF and PACF
    plot_acf(data['value'].dropna())
    plot_pacf(data['value'].dropna())
    plt.show()

    # Fit ARIMA model (replace p, d, q with identified values)
    model = ARIMA(data['value'].dropna(), order=(p, d, q))  # Replace with actual values
    model_fit = model.fit()

    # Summary of model
    print(model_fit.summary())

    # Forecasting future values
    forecast = model_fit.forecast(steps=10)  # Forecast next 10 periods
    print(forecast)

# Apply ARIMA to each dataset
apply_arima('BOE-XUDLERD.csv')
apply_arima('monthly-milk-production.csv')
apply_arima('Time_Series_Temp_Data.csv')

Conclusion
Applying the ARIMA model to the BOE-XUDLERD, monthly milk production, and temperature datasets allows for effective forecasting based on historical trends. By following a structured approach—checking for stationarity, identifying parameters, fitting the model, and evaluating its performance—you can derive valuable insights from your time series data. This methodology is crucial for making informed decisions in various fields such as finance, agriculture, and climate science.

