import numpy as np
from datetime import timedelta

def make_predictions(model, X_train, X_test, y_train, y_test, scaler, close_scaler):
    train_pred = model.predict(X_train, verbose=0)
    test_pred  = model.predict(X_test,  verbose=0)

    # Inverse transform using close_scaler (single feature)
    train_pred = close_scaler.inverse_transform(train_pred)
    test_pred  = close_scaler.inverse_transform(test_pred)

    y_train_actual = close_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual  = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(np.mean((test_pred - y_test_actual) ** 2))
    total_actual = np.concatenate((y_train_actual, y_test_actual))

    return train_pred, test_pred, total_actual, rmse


def next_day_prediction(model, scaled_data, close_scaler, window_size=60):
    last_window = scaled_data[-window_size:].reshape(1, window_size, scaled_data.shape[1])
    next_scaled = model.predict(last_window, verbose=0)
    return close_scaler.inverse_transform(next_scaled)[0][0]


def forecast_n_days(model, scaled_data, close_scaler, last_date, window_size=60, forecast_days=5):
    n_features = scaled_data.shape[1]
    current_batch = scaled_data[-window_size:].copy()  # (window, n_features)
    forecast_scaled = []

    for _ in range(forecast_days):
        inp = current_batch.reshape(1, window_size, n_features)
        next_pred = model.predict(inp, verbose=0)[0][0]  # scaled close
        forecast_scaled.append([[next_pred]])

        # Shift window: drop oldest row, append new row
        # For new row: use last row's features but update Close (index 0)
        new_row = current_batch[-1].copy()
        new_row[0] = next_pred
        current_batch = np.vstack([current_batch[1:], new_row])

    forecast_prices = close_scaler.inverse_transform(
        np.array([f[0] for f in forecast_scaled])
    )
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    return forecast_dates, forecast_prices