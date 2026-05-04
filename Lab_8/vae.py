from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_dim = 20
latent_dim = 2

inputs = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(inputs)
latent = Dense(latent_dim)(encoded)

decoded = Dense(10, activation='relu')(latent)
outputs = Dense(input_dim, activation='sigmoid')(decoded)

vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse')
