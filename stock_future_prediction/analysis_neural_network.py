import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


# TODO: FIX THE DATA LEAKAGE!!!

def interactive_plot(df, title):
  fig = px.line(title=title)
  for i in df.columns[1:]:
    fig.add_scatter(x=df.Date, y=df[i], name=i)
  fig.show()


# concatenate the date, stock price, and the volume data in 1 DataFrame object
# name is the enum of stock tickers i.e 'AAPL' passed as a string
def individual_stock(price_df, vol_df, name):
  return pd.DataFrame({'Date': price_df.Date, 'Close': price_df[name], 'Volume': vol_df[name]})


# read and sort the stock prices data
stock_prices = pd.read_csv('../data/stocks.csv')
stock_prices = stock_prices.sort_values(by=['Date'])

# read and sort the stock volumes data - its the trades volume
stock_volumes = pd.read_csv('../data/stock_volume.csv')
stock_volumes = stock_volumes.sort_values(by=['Date'])

# check for null values
if stock_prices.isnull().sum().any():
  raise RuntimeError('Missing Data in Stock Prices')
if stock_volumes.isnull().sum().any():
  raise RuntimeError('Missing Data in Stock Volumes');

price_volume_df = individual_stock(stock_prices, stock_volumes, 'sp500')

# get the close and volume data as training data
training_data = price_volume_df.iloc[:, 1:3].values

# Normalise the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_data)

# create the training and testing data
# training data -> present data and previous dat values
# Y -> the target price, price tomorrow
X = []
Y = []
for i in range(1, len(price_volume_df)):
  X.append(training_set_scaled[i - 1:i, 0])
  Y.append(training_set_scaled[i, 0])

# convert the data into numpy arrays
X = np.asarray(X)
Y = np.asarray(Y)

# Data Splitting
split = int(0.7 * len(X))
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]

# Reshape the 1D arrays to 3D arrays to feed in the model
# add an additional dimension
# X_train + X_test == total number of samples
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Keras API is used to create the model
# X_train.shape[0] -> the size of the training data
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
# add additional layers
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)

# specify the activation function in the output
# output will be a continuous value -> use linear
outputs = keras.layers.Dense(1, activation='linear')(x)

# build the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
# by specifying the type of optimiser
model.compile(optimizer='adam', loss='mse')

# get the model summary
print(model.summary())

# train the model
# validation split -> cross check to avoid over fitting
# epochs -> number of outer loops, iteration
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2)

# Make predictions after the model is trained
predicted = model.predict(X)

# store the predictions in a python list
test_predicted = []

for i in predicted:
  # flatten before adding to the list
  test_predicted.append(i[0][0])

# create a new dataframe with the predicted data
df_predicted = price_volume_df[1:][['Date']]

# add the predictions as a column to the dataframe
df_predicted['Predictions'] = test_predicted

close = []
for i in training_set_scaled:
  close.append(i[0])

df_predicted['Close'] = close[1:]

print(df_predicted)

interactive_plot(df_predicted, 'Actual vs LSTM Predictions')
