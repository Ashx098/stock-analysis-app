import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit UI
st.title("📈 Stock Analysis & Prediction System")
st.write("Enter a stock ticker and click 'Predict' to get started!")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "NVDA")

if st.button("Predict Stock Price"):
    st.write(f"Fetching stock data for {ticker}...")

    # Fetch stock data
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")

    if df.empty:
        st.error("❌ Invalid Stock Ticker! Please enter a valid stock symbol.")
    else:
        # Keep only the 'Close' column
        df = df[['Close']].dropna()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        df_scaled = scaler.fit_transform(df)

        # Create sequences for LSTM
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        seq_length = 60
        X, y = create_sequences(df_scaled, seq_length)

        # Split into train & test sets (80% train, 20% test)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.3),
            LSTM(25, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation="relu"),
            Dense(1)
        ])

        # Compile Model
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train Model
        with st.spinner("Training LSTM Model..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # Make Predictions
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate RMSE
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

        st.write(f"📊 **Model Accuracy: RMSE = {rmse:.2f}**")

        # Plot Predictions
        plt.figure(figsize=(12,6))
        plt.plot(y_test_rescaled, label="Actual Stock Price", color="blue")
        plt.plot(y_pred_rescaled, label="Predicted Stock Price", color="red")
        plt.legend()
        plt.title(f"{ticker} Stock Price Prediction (LSTM)")
        st.pyplot(plt)
