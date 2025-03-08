import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Streamlit UI
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")

st.title("📈 Stock Analysis & Prediction System")
st.write("Select a stock to view trends and future predictions.")

# ✅ List of Top Tech Stocks for Quick Selection
tech_stocks = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA"]
selected_stock = st.radio("Select a Stock:", tech_stocks, horizontal=True)

# ✅ Slider to Select Prediction Days
future_days = st.slider("Select Future Prediction Days:", 1, 7, 5)

# Fetch Stock Data Automatically
st.write(f"Fetching stock data for {selected_stock}...")

# Fetch stock data
stock = yf.Ticker(selected_stock)
df = stock.history(period="5y")

if df.empty:
    st.error("❌ Invalid Stock Ticker! Please choose a valid stock.")
else:
    # Keep relevant columns
    df = df[['Close']].dropna().reset_index()

    # Display Data
    st.subheader(f"{selected_stock} Historical Stock Data")
    st.dataframe(df.tail(10))  # Show last 10 days

    # ✅ Interactive Graph for Stock Prices
    st.subheader(f"{selected_stock} Stock Price Trend (Interactive)")
    fig = px.line(df, x='Date', y='Close', title=f"{selected_stock} Stock Price Trend")
    st.plotly_chart(fig)

    # ✅ Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[['Close']])

    # ✅ Function to Create Sequences
    def create_sequences(data, seq_length=30):  # Using 30 days sequence for better trend learning
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # ✅ Prepare LSTM sequences
    seq_length = 30  # Using last 30 days to predict the next prices
    X, y = create_sequences(df_scaled, seq_length)

    # Train/Test Split (80-20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # ✅ Caching LSTM Model Training to Speed Up App
    @st.cache_resource
    def train_lstm_model():
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),  # Increased LSTM units
            Dropout(0.3),  # Lower dropout to retain more learning
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation="relu"),
            Dense(1)
        ])

        # ✅ Use Adam Optimizer with Learning Rate Decay
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        # ✅ Train with Early Stopping to Prevent Overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

        return model

    model = train_lstm_model()  # Train & Cache Model

    # ✅ Predict Future Prices
    future_prices = []

    # Use last sequence for prediction
    last_sequence = df_scaled[-seq_length:].reshape(1, seq_length, 1)

    for _ in range(future_days):  # Predict only selected days
        next_price = model.predict(last_sequence)[0][0]
        future_prices.append(next_price)
        last_sequence = np.append(last_sequence[:,1:,:], [[[next_price]]], axis=1)

    # Rescale Predictions Back to Original Price
    future_prices_rescaled = scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

    # Create Future Dates
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=future_days+1, freq="D")[1:]

    # ✅ Calculate RMSE
    y_pred_rescaled = scaler.inverse_transform(model.predict(X_test))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    st.write(f"📊 **Model Accuracy: RMSE = {rmse:.2f}**")

    # ✅ Plot Training & Validation Loss
    st.subheader("📉 LSTM Training Loss")
    loss_fig, loss_ax = plt.subplots(figsize=(8, 4))
    loss_ax.plot(model.history.history["loss"], label="Training Loss")
    loss_ax.plot(model.history.history["val_loss"], label="Validation Loss")
    loss_ax.legend()
    loss_ax.set_title("LSTM Training Loss (Optimized)")
    st.pyplot(loss_fig)

    # ✅ Plot Predictions
    st.subheader(f"📈 {selected_stock} Future Stock Price Prediction")
    pred_fig, pred_ax = plt.subplots(figsize=(12, 6))
    pred_ax.plot(df['Date'], df['Close'], label="Historical Prices", color="blue")
    pred_ax.plot(future_dates, future_prices_rescaled, 'ro', label="Predicted Prices")  # Plot future predictions as red dots
    pred_ax.set_title(f"{selected_stock} Future Stock Price Prediction (LSTM)")
    pred_ax.set_xlabel("Date")
    pred_ax.set_ylabel("Stock Price (USD)")
    pred_ax.legend()
    pred_ax.grid()
    st.pyplot(pred_fig)

    # ✅ Disclaimer
    st.info("⚠ **Note:** Stock market predictions are inherently uncertain. Please use this tool for analysis, not financial decisions.")
