import numpy as np
from data_loader import load_data
from preprocessing import scale_data, create_sequences
from model import build_model
from train import train_model
from predict import make_predictions, next_day_prediction, forecast_5_days
from visualize import plot_initial_trend, plot_predictions, plot_5day_forecast

print("Tip: For Indian stocks add .NS (e.g., RELIANCE.NS)")
stock_name = input("Enter Stock Ticker: ").upper()

data = load_data(stock_name)

if data.empty:
    print("No data found.")
    exit()

print(f"Data downloaded up to: {data.index[-1]}")

plot_initial_trend(data, stock_name)

close_prices, scaled_data, train_data, test_data, scaler, training_data_len = scale_data(data)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = build_model((X_train.shape[1], 1))

history = train_model(model, X_train, y_train)

train_pred, test_pred, total_actual, rmse = make_predictions(
    model, X_train, X_test, y_train, y_test, scaler
)

print(f"Test RMSE: {rmse:.2f}")

plot_predictions(total_actual, train_pred, test_pred, stock_name)

next_price = next_day_prediction(model, scaled_data, scaler)
current_price = close_prices[-1][0]

print(f"\nCurrent Price: ₹{current_price:.2f}")
print(f"Predicted Tomorrow: ₹{next_price:.2f}")

forecast_dates, forecast_prices = forecast_5_days(
    model, scaled_data, scaler, data.index[-1]
)

for date, price in zip(forecast_dates, forecast_prices):
    print(f"{date.strftime('%Y-%m-%d')}: ₹{price[0]:.2f}")

plot_5day_forecast(data, forecast_dates, forecast_prices, stock_name)
