from model import get_callbacks

def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    callbacks = get_callbacks()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
        validation_split=0.1,   # 10% of training for val loss monitoring
    )
    return history