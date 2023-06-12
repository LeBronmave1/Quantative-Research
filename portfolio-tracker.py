import yfinance as yf
import pandas as pd

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Calculate moving average of a given stock
def calculate_moving_average(data, window):
    data['MA'] = data['Close'].rolling(window=window).mean()

# Perform a simple moving average crossover trading strategy
def perform_strategy(data):
    data['Position'] = 0

    for i in range(1, len(data)):
        if data['MA'].iloc[i] > data['MA'].iloc[i-1]:
            data['Position'].iloc[i] = 1
        else:
            data['Position'].iloc[i] = -1

    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']

    return data

def main():
    # Define the stock symbol and date range
    symbol = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2023-06-10'

    # Fetch the stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Calculate the 50-day moving average
    calculate_moving_average(stock_data, 50)

    # Calculate the daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # Perform the trading strategy
    strategy_data = perform_strategy(stock_data)

    # Print the strategy data
    print(strategy_data)

if __name__ == "__main__":
    main()
