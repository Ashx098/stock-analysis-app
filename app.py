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

# Initialize session state for button clicks
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

# Streamlit UI
st.title("📈 Stock Analysis & Prediction System")
st.write("Select a stock and click 'Fetch Data' to get started!")

# List of top tech stocks
tech_stocks = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA"]
selected_stock = st.selectbox("Select a Stock:", tech_stocks)

if st.button("Fetch Data"):
    st.session_state.predict_clicked = False  # Reset prediction state
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

        # Plot Interactive Graph
        st.subheader(f"{selected_stock} Stock Price Trend (Interactive)")
        fig = px.line(df, x='Date', y='Close', title=f"{selected_stock} Stock Price Trend")
        st.plotly_chart(fig)

        # Future Prediction Button
        if st.button("Predict Future Trend"):
            st.session_state.predict_clicked = True  # Store button state

# Run only if Predict Future Trend was clicked
if st.session_state.predict_clicked:
    st.subheader(f"🔮 Predicting Future Prices for {selected_stock}")

    # Fetch stock data again (to persist after rerun)
    stock = yf.Ticker(selected_stock)
    df = stock.history(period="5y")[['Close']].dropna().reset_index()

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[['Close']])

    # Function to create sequences
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Prepare LSTM sequences
    seq_length = 60
    X, y = create_sequences(df_scaled, seq_length)

    # Train/Test Split (80-20)
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
    with st.spinner("🚀 Training LSTM Model... Please wait"):
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict Future Prices
    future_days = 180  # Predict next 6 months
    future_prices = []

    # Use last sequence for prediction
    last_sequence = df_scaled[-seq_length:].reshape(1, seq_length, 1)

    for _ in range(future_days):
        next_price = model.predict(last_sequence)[0][0]
        future_prices.append(next_price)
        last_sequence = np.append(last_sequence[:,1:,:], [[[next_price]]], axis=1)

    # Rescale predictions back to original price
    future_prices_rescaled = scaler.inverse_transform(np.array(future_prices).reshape(-1,1))

    # Create future dates
    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=future_days+1, freq="D")[1:]

    # Calculate RMSE
    from sklearn.metrics import mean_squared_error
    y_pred_rescaled = scaler.inverse_transform(model.predict(X_test))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    st.write(f"📊 **Model Accuracy: RMSE = {rmse:.2f}**")

    # Plot Predictions
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label="Historical Prices", color="blue")
    plt.plot(future_dates, future_prices_rescaled, label="Predicted Prices", color="red")
    plt.title(f"{selected_stock} Future Stock Price Prediction (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
