from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

def build_model(model_type):
    model = Sequential()

    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(10, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(10, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(10, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
