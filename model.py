from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, SimpleRNN, LSTM, GRU,
    Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_model(input_shape, model_type="LSTM", units=64, dropout=0.2):
    """
    Improved model — deeper, BatchNorm, better optimizer.
    input_shape: (window_size, n_features)
    """
    model = Sequential()

    layer_map = {
        "SimpleRNN": SimpleRNN,
        "LSTM":      LSTM,
        "GRU":       GRU,
    }
    Layer = layer_map.get(model_type, LSTM)

    if model_type == "LSTM":
        # Bidirectional LSTM — sees past AND future context in window
        model.add(Bidirectional(
            LSTM(units, return_sequences=True),
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Bidirectional(LSTM(units // 2, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(LSTM(units // 4))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(GRU(units // 2, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(GRU(units // 4))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    else:  # SimpleRNN
        model.add(SimpleRNN(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(units // 2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(units // 4))
        model.add(Dropout(dropout))

    # Dense head
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    # Lower LR + clipnorm for stability
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='huber')  # Huber = less sensitive to outliers

    return model

def get_callbacks():
    """Early stopping + LR reduction to avoid overfitting."""
    return [
        EarlyStopping(
            monitor='loss', patience=5,
            restore_best_weights=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor='loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=0
        ),
    ]