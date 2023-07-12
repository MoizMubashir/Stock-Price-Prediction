# Importing the libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import keras
import tensorflow as tf
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st
import yfinance as yf

# Defining the ticker symbol for the stock
ticker = "AAPL"

st.title("Stock Trend Prediction")

# Defining the start and end dates
start_date = "1999-01-01"
# Get the current date
current_date = datetime.today().strftime('%Y-%m-%d')
end_date = current_date 

user_input = st.text_input('Enter Stock Ticker', ticker)
# Fetching the historical stock data for the specified date range
df = yf.download(user_input, start=start_date, end=end_date)

# Describing Data
st.subheader('Data From 2010 - 2019')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(16,8))
plt.plot(df.Close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig)

st.subheader('Opening Price | Closing Price | High Price | Low Price')
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16,8), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(df['Open'], color='red', label = 'Opening Price')
plt.plot(df['Close'], color='green', label = 'Closing Price')
plt.plot(df['Low'], color='black', label = 'Low Price')
plt.plot(df['High'], color='blue', label = 'High Price')
plt.legend(loc='best')
st.pyplot(fig)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

ax1.plot(df['Open'], color='red')
ax1.set_title('Opening Price')

ax2.plot(df['Close'], color='green')
ax2.set_title('Closing Price')

ax3.plot(df['Low'], color='black')
ax3.set_title('Low Price')

ax4.plot(df['High'], color='blue')
ax4.set_title('High Price')

# Displaying the subplots in Streamlit
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(16,8))
plt.plot(ma100, 'r')
plt.plot(df.Close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(['ma100', 'Close'], loc='lower right')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(16,8))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(['ma100', 'ma200','close'], loc='lower right')
st.pyplot(fig)

#----------------------------------------------------
# Create a new dataframe with only the 'Close' column
data = df[['Close']].copy()

dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Load the model 
model = load_model('keras_model3.h5')

# Create the testing data set
# Create a new array containing scaled values from index 4868 to 6160
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] 
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0]) 


x_test = np.array(x_test)

# Reshaping the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # (number of samples, number of timesteps, number of features "which is the close price")

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Getting Predictions
st.subheader("Prediction Price vs Actual Price")
# Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualizing the data
fig = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig)


ticker_quote = yf.download(user_input, start=start_date, end=end_date)
new_df = ticker_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
st.write("Predicted price of the stock: ")
st.write(pred_price)
st.write("at")
st.write(end_date)

# Get the quote
# Define the start and end dates
st.write('Stock Price')
start_date_1 = st.text_input('Enter the Start Date of predictions("yyyy-mm-dd"): ')
# start_date_1 = "2023-06-29"
end_date_1 = st.text_input('Enter End Date of predictions("yyyy-mm-dd"): ')
# end_date_1 = "2023-07-11"
ticker_quote2 = yf.download(user_input, start=start_date_1, end=end_date_1)
st.write("Actual Price of Stock")
st.write(ticker_quote2['Close'])

