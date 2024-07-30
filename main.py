import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib.dates import DateFormatter, date2num
import pandas as pd

# Currently would only provide a reasonable prediction for cyclical stocks - CMRE DAC

# ---------------------------------------------------------------------------- #

stock_data = yf.download('DAC', start='2014-07-26', end='2024-07-26') # Enter stock ticker symbol and date here

#scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], time_step, 1)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=64))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1) # Enter epochs amount here

test_loss = model.evaluate(X_test, y_test, verbose=1)
print('Test Loss:', test_loss)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

future_days = 60
last_date = stock_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

future_data = scaled_data[-time_step:]
future_X = []
for i in range(future_days):
    future_X.append(future_data)
    next_pred = model.predict(future_data.reshape((1, time_step, 1)))
    future_data = np.append(future_data[1:], next_pred, axis=0)

future_X = np.array(future_X)
future_X = future_X.reshape((future_X.shape[0], future_X.shape[1], 1))

future_predictions = model.predict(future_X)
future_predictions = scaler.inverse_transform(future_predictions)

highest_point = np.max(future_predictions)
lowest_point = np.min(future_predictions)
highest_date = future_dates[np.argmax(future_predictions)]
lowest_date = future_dates[np.argmin(future_predictions)]

original_data = stock_data['Close'].values
plot_data = np.append(original_data, [np.nan] * future_days)
predicted_data = np.empty(len(plot_data))
predicted_data[:] = np.nan
predicted_data[len(original_data) - len(predictions):len(original_data)] = predictions.flatten()
predicted_data[len(original_data):] = future_predictions.flatten()

plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, original_data, label='Original Data')
plt.plot(stock_data.index[-len(predictions):], predicted_data[-len(predictions):], label='Predicted Data')

future_dates_num = date2num(future_dates)
plt.plot(future_dates_num, future_predictions, label='Future Predictions', color='red')

plt.scatter([highest_date], [highest_point], color='green', label='Highest Point', zorder=5)
plt.scatter([lowest_date], [lowest_point], color='blue', label='Lowest Point', zorder=5)
plt.annotate(f'Highest: {highest_point:.2f}\nDate: {highest_date.date()}', 
             (highest_date, highest_point), 
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center', 
             color='black')
plt.annotate(f'Lowest: {lowest_point:.2f}\nDate: {lowest_date.date()}', 
             (lowest_date, lowest_point), 
             textcoords="offset points", 
             xytext=(0,-15), 
             ha='center', 
             color='black')

date_form = DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(date_form)
plt.gcf().autofmt_xdate(rotation=45)
plt.legend()

plt.show()
