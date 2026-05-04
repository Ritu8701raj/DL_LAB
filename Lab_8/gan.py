from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

generator = Sequential([
    Dense(16, activation='relu', input_dim=10),
    Dense(1, activation='sigmoid')
])

discriminator = Sequential([
    Dense(16, activation='relu', input_dim=1),
    Dense(1, activation='sigmoid')
])
