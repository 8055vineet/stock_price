import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import speech_recognition as sr  # For voice input

# Set environment variable to suppress oneDNN warnings (if needed)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
model = load_model(r"C:\Users\kantl\Desktop\app\Stock Price Prediction Model.keras")

# App title
st.markdown('# Stock Price Analysis')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2013, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2024, 4, 10))

# Function to take voice input and convert to text
def take_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.sidebar.write(f"Recognized Company: {text}")
            return text
        except sr.UnknownValueError:
            st.sidebar.write("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.sidebar.write("Sorry, could not request results; check your internet connection.")
    return None

# Voice Button to get the company name
if st.sidebar.button("Use Voice Input"):
    company_name = take_voice_input()
    if company_name:
        # Use voice input to search the company symbol
        ticker_list = pd.read_csv(r"C:\Users\kantl\Desktop\app\data.csv")
        matched_ticker = ticker_list[ticker_list['Name'].str.contains(company_name, case=False, na=False)]
        if not matched_ticker.empty:
            tickerSymbol = matched_ticker.iloc[0]['Symbol'] + " - " + matched_ticker.iloc[0]['Name']
        else:
            st.sidebar.write("No matching company found. Please try again.")
else:
    # Normal selection of ticker symbol from dropdown
    ticker_list = pd.read_csv(r"C:\Users\kantl\Desktop\app\data.csv")
    tickerSymbol = st.sidebar.selectbox('Stock Symbol', ticker_list['Symbol'] + " - " + ticker_list['Name'])

# Retrieve stock data
tickerData = yf.Ticker(tickerSymbol.split(" ")[0])
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# Display logo
def remove_prefix(url, prefix):
    if url.startswith(prefix):
        return url[len(prefix):]
    return url

prefix = "http://"
prefix2 = "http://www."
link = tickerData.info['website']
logo_link = remove_prefix(link, prefix2)
if link == logo_link:
    logo_link = remove_prefix(link, prefix)
logo_name = f"https://logo.clearbit.com/{logo_link}"
st.markdown(f'<img src={logo_name}>', unsafe_allow_html=True)

# Display stock info
string_name = f"{tickerData.info['longName']} ({tickerSymbol.split(' ')[0]})"
st.header(f'**{string_name}**')
st.info(tickerData.info['longBusinessSummary'])

# Display stock data
st.header('**Stock Data**')
st.write(tickerDf.style.set_table_attributes("style='width: 700px; height: 400px;'"))

# Visualization - Bollinger Bands
st.header('**Bollinger Bands**')
qf = cf.QuantFig(tickerDf, title='First Quant Figure', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

# Price history and moving averages
def plot_moving_averages(df, days_list, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Close'], 'g', label='Actual Value')
    for days in days_list:
        ma = df['Close'].rolling(days).mean()
        ax.plot(ma, label=f'MA{days}')
    ax.set_title(title)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Close Price USD ($)", fontsize=18)
    ax.legend(loc='upper left')
    st.pyplot(fig)

plot_moving_averages(tickerDf, [50], 'Price vs MA50')
plot_moving_averages(tickerDf, [50, 100], 'Price vs MA50 vs MA100')
plot_moving_averages(tickerDf, [100, 200], 'Price vs MA100 vs MA200')

# Model prediction preparation
data = tickerDf.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create testing data
test_data = scaled_data[training_data_len - 100:, :]
x_test = [test_data[i - 100:i, 0] for i in range(100, len(test_data))]
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# RMSE calculation
rmse = np.sqrt(np.mean((predictions - dataset[training_data_len:]) ** 2))
st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(dataset[training_data_len:], predictions)
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot true vs predicted prices
st.subheader('True vs Predicted Stock Prices')
fig5 = plt.figure(figsize=(16, 8))
plt.plot(dataset[training_data_len:], label='True Prices', marker='o')
plt.plot(predictions, label='Predicted Prices', marker='x')
plt.xlabel('Days', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.title('True vs Predicted Stock Prices')
plt.legend()
st.pyplot(fig5)

# Plot MAPE
st.subheader('Mean Absolute Percentage Error (MAPE)')
fig6 = plt.figure(figsize=(16, 8))
plt.plot(np.arange(1, len(dataset[training_data_len:]) + 1),
         np.abs((np.array(dataset[training_data_len:]) - np.array(predictions)) / np.array(dataset[training_data_len:])) * 100,
         label='MAPE', marker='o', color='green')
plt.xlabel('Days', fontsize=18)
plt.ylabel('MAPE (%)', fontsize=18)
plt.title('Mean Absolute Percentage Error (MAPE)')
plt.legend()
st.pyplot(fig6)

# Predictions using the model
last_60_days = dataset[-60:].reshape(-1, 1)  # Ensure the shape is correct
last_60_days_scaled = scaler.transform(last_60_days)
X_test = np.array([last_60_days_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price = pred_price[0][0]
st.sidebar.write(f"Predicted Stock Price : $ {pred_price:.6f}")

st.write('---')
st.write("""**Credits**
 - App built by Vineet Patel (21U02020)""")
