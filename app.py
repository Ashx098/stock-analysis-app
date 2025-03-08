import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Streamlit App Title
st.title("📈 Stock Analysis & Prediction System")

# User input for stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "NVDA")

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch last 5 years of data
    return df

# Fetch data when button is clicked
if st.button("Fetch Data"):
    stock_data = get_stock_data(stock_ticker)

    if not stock_data.empty:
        st.subheader(f"Stock Data for {stock_ticker.upper()}")
        st.write(stock_data.tail())  # Show last few rows
        
        # Calculate Moving Averages
        stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()  # 50-day SMA
        stock_data["EMA_20"] = stock_data["Close"].ewm(span=20, adjust=False).mean()  # 20-day EMA
        
        # Interactive Plotly Chart
        st.subheader("📊 Stock Price Chart with Moving Averages")
        fig = go.Figure()
        
        # Add Closing Price Line
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode="lines", name="Closing Price"))
        
        # Add 50-day Simple Moving Average
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], mode="lines", name="50-day SMA", line=dict(dash="dash")))
        
        # Add 20-day Exponential Moving Average
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["EMA_20"], mode="lines", name="20-day EMA", line=dict(dash="dot")))
        
        # Layout Settings
        fig.update_layout(title=f"{stock_ticker.upper()} Price Chart",
                          xaxis_title="Date",
                          yaxis_title="Stock Price (USD)",
                          legend_title="Legend",
                          template="plotly_dark")
        
        # Display Chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # -----------------
        # 📌 Prophet Model for Forecasting (Fixed Version)
        # -----------------
        st.subheader("📈 Stock Price Prediction (Prophet Model)")

        # Prepare Data for Prophet
        df_prophet = stock_data.reset_index()[["Date", "Close"]]
        df_prophet.columns = ["ds", "y"]  # Prophet expects "ds" (date) and "y" (value)

        # Fix: Remove Timezone from Date
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

        # Fix: Remove missing values
        df_prophet = df_prophet.dropna()

        # Fix: Ensure "y" column is numeric
        df_prophet = df_prophet[df_prophet["y"].apply(lambda x: isinstance(x, (int, float)))]

        # Check if data is valid for Prophet
        if len(df_prophet) < 10:
            st.error("Not enough data for prediction. Try another stock.")
        else:
            # Train Prophet Model
            model = Prophet()
            model.fit(df_prophet)

            # Predict for next 180 days
            future = model.make_future_dataframe(periods=180)  # Forecast for 6 months
            forecast = model.predict(future)

            # Plot Forecast
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted Price"))
            fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot")))
            fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot")))

            fig_forecast.update_layout(title=f"Stock Price Prediction for {stock_ticker.upper()}",
                                       xaxis_title="Date",
                                       yaxis_title="Predicted Price (USD)",
                                       template="plotly_dark")

            st.plotly_chart(fig_forecast, use_container_width=True)

    else:
        st.error("Failed to retrieve stock data. Please check the ticker.")

st.write("👆 Enter a stock ticker and click 'Fetch Data' to get started!")
'''
'''