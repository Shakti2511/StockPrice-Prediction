import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
# Sequential will help in to intialize the neural network
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from joblib import dump


scaler = MinMaxScaler(feature_range=(0,1))

df = pd.read_csv("TSLA.csv")
print(df.head())                       # By default it will print 5 rows of data
print(df.shape)                       # It will print the number of rows and number of columns

df["Date"] = pd.to_datetime(df.Date, format = "%Y-%m-%d")           # to_datetime() will convert the date in a certain format
df.index = df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"])
plt.show()


data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data['Close'][i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

final_dataset = new_dataset.values


part_values = int(0.75 * len(final_dataset))
train_data = final_dataset[0:part_values,:]
valid_data = final_dataset[part_values:,:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for x in range(120, len(train_data)):
    x_train_data.append(scaled_data[x-120:x, 0])
    y_train_data.append(scaled_data[x, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dense(1))

print("Creating the model")

lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
lstm_model.fit(x_train_data, y_train_data, epochs = 100, batch_size = 32, verbose = 2)


# Save the scaler
dump(scaler, 'scaler.joblib')

lstm_model.save("Tsla.h5")
print("model saved")