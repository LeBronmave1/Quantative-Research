import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Data collection
# Assuming you have a CSV file 'stock_prices.csv' containing historical stock prices
df = pd.read_csv('stock_prices.csv')

# Step 2: Data preprocessing
# Assuming the dataset has 'Date' and 'Close' columns
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# Visualize the time series data
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Historical Stock Prices')
plt.show()

# Step 4: Feature engineering (optional)
# Here, we'll use the original closing prices as the feature

# Step 5: Model selection
# Assuming you've determined that ARIMA is a suitable model
# ARIMA(p, d, q) - p: AR order, d: differencing order, q: MA order

# Splitting the data into train and test sets
train_data = df['Close'].iloc[:int(0.8*len(df))]
test_data = df['Close'].iloc[int(0.8*len(df)):]

# Step 6: Model training and validation
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# Step 7: Model evaluation
# Forecasting the stock prices for the test set
predictions = model_fit.forecast(steps=len(test_data))[0]

# Calculate evaluation metrics
mae = np.mean(np.abs(predictions - test_data))
mse = np.mean((predictions - test_data) ** 2)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Step 8: Model deployment
# Assuming you want to visualize the predictions
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()