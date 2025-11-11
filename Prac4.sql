# ============================
# Practical 4: Anomaly Detection using Autoencoder (LSTM example from PDF)
# Colab-ready: Paste into one notebook cell and run
# ============================


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

csv_path = '/content/GOOG.csv'
df = pd.read_csv(csv_path)
print("Loaded CSV shape:", df.shape)
display(df.head())


df = df[['Date', 'Close']]
print("\nColumns after selecting Date and Close:")
display(df.head())
df.info()


print("\nDate range in dataset:", df['Date'].min(), "to", df['Date'].max())
train = df.loc[df['Date'] <= '2017-12-24'].copy()
test  = df.loc[df['Date'] > '2017-12-24'].copy()
print("Train shape:", train.shape, "Test shape:", test.shape)


scaler = StandardScaler()
scaler = scaler.fit(np.array(train['Close']).reshape(-1,1))

train['Close'] = scaler.transform(np.array(train['Close']).reshape(-1,1))
test['Close']  = scaler.transform(np.array(test['Close']).reshape(-1,1))


plt.figure(figsize=(10,4))
plt.plot(train['Date'], train['Close'], label='scaled - train')
plt.xticks([], [])
plt.legend()
plt.title('Scaled Close (Train)')
plt.show()


TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    X_out, y_out = [], []
    for i in range(len(X) - time_steps):
        X_out.append(X.iloc[i:(i+time_steps)].values)
        y_out.append(y.iloc[i+time_steps])
    return np.array(X_out), np.array(y_out)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test,  y_test  = create_sequences(test[['Close']], test['Close'])

print("Training input shape: ", X_train.shape)
print("Testing  input shape: ", X_test.shape)


print("\nExample sequence (last training sample):")
print(X_train[-1].reshape(-1))


np.random.seed(21)
tf.random.set_seed(21)


model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()


history = model.fit(
    X_train,
    X_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
    shuffle=False,
    verbose=2
)


plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1,2))

plt.figure(figsize=(8,4))
plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')
plt.title('Histogram of training reconstruction error')
plt.show()


threshold = np.max(train_mae_loss)
print('Reconstruction error threshold (max):', threshold)


X_test_pred = model.predict(X_test, verbose=1)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=(1,2))

plt.figure(figsize=(8,4))
plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')
plt.title('Histogram of test reconstruction error')
plt.show()


anomaly_df = pd.DataFrame(test.iloc[TIME_STEPS:].reset_index(drop=True).copy())
anomaly_df['loss'] = test_mae_loss
anomaly_df['threshold'] = threshold
anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']

anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]
print("\nNumber of anomalies detected in test set:", anomalies.shape[0])
display(anomalies.head())


fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=scaler.inverse_transform(anomaly_df['Close'].values.reshape(-1,1)).flatten(),
                         name='Close price'))
fig.add_trace(go.Scatter(x=anomalies['Date'],
                         y=scaler.inverse_transform(anomalies['Close'].values.reshape(-1,1)).flatten(),
                         mode='markers', name='Anomaly', marker=dict(color='red', size=6)))
fig.update_layout(showlegend=True, title='Detected anomalies (Autoencoder based)')
fig.show()


