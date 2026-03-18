# TradeVision AI — RNN Stock Predictor

TradeVision AI is a professional Streamlit application that uses Recurrent Neural Networks (RNN) to forecast stock prices.

## 🚀 Features
- **Advanced RNN Predictions**: Uses 3-layer SimpleRNN architecture.
- **Dynamic Hyperparameters**: Live adjust Epochs, Window Size, and Forecast Horizon.
- **Interactive Technical Analysis**: RSI, Bollinger Bands, and Volume charts.
- **Model Performance Metrics**: RMSE, Residual tracking, and Actual vs Predicted scatter plots.
- **Financial Insights**: Fundamental data, News headlines, and Analyst recommendations.

## 🛠️ Setup
1. Create a virtual environment:
   ```bash
   python -m venv tv_env
   source tv_env/bin/activate  # On Windows: tv_env\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📄 Project Structure
- `app.py`: Main Streamlit application and UI logic.
- `data_loader.py`: Fetches market data using `yfinance`.
- `preprocessing.py`: Data scaling and sequence creation.
- `model.py`: RNN model architecture.
- `train.py`: Training logic.
- `predict.py`: Prediction and forecasting functions.
- `requirements.txt`: Python package dependencies.
