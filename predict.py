import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to create sequences
def create_sequences(data_X, data_y, sequence_length):
    x = []
    y = []
    for i in range(sequence_length, len(data_X)):
        x.append(data_X[i - sequence_length:i])
        y.append(data_y[i])
    return np.array(x), np.array(y)

# Fetch historical Bitcoin prices
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
params = {'vs_currency': 'usd', 'days': '365'}  # Fetch one year of data
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    if 'prices' in data:
        prices = data['prices']
    else:
        print("Error: 'prices' key not found in the API response.")
        print("API response:", data)
        exit()
else:
    print(f"Error fetching data. Status code: {response.status_code}")
    print("API response:", data)
    exit()

# Convert data to DataFrame
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
df.drop('timestamp', axis=1, inplace=True)

# Feature engineering
df['returns'] = df['price'].pct_change()
df['ma7'] = df['price'].rolling(window=7).mean()
df['ma21'] = df['price'].rolling(window=21).mean()
df['stddev'] = df['price'].rolling(window=21).std()
df['upper_band'] = df['ma21'] + (df['stddev'] * 2)
df['lower_band'] = df['ma21'] - (df['stddev'] * 2)
df['ema'] = df['price'].ewm(com=0.5).mean()
df['momentum'] = df['price'] - df['price'].shift(1)
df.dropna(inplace=True)

# Prepare the dataset
features = ['returns', 'ma7', 'ma21', 'stddev', 'upper_band', 'lower_band', 'ema', 'momentum']
X = df[features].values
y = df['price'].values.reshape(-1, 1)

# Initialize separate scalers
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the data
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Create sequences
sequence_length = 60  # Using past 60 days to predict the next day
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

# Split the data
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build the improved LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=32)))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# Compile the model with adjusted learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Implement early stopping and reduce learning rate on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model with more epochs and validation split
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot training & validation loss
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
predictions_scaled = model.predict(X_test)

# Inverse transform the predictions
predictions_inverse = scaler_y.inverse_transform(predictions_scaled)

# Inverse transform the actual values
y_test_inverse = scaler_y.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_inverse, predictions_inverse)
mae = mean_absolute_error(y_test_inverse, predictions_inverse)
rmse = np.sqrt(mse)
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')