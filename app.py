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
    The code above is a Streamlit app that fetches stock data using the  yfinance  library, calculates moving averages, and displays an interactive chart using Plotly. It also uses the  Prophet  library to predict stock prices for the next 180 days. 
    To run the Streamlit app, save the code to a file named  app.py  and run the following command in the terminal: 
    streamlit run app.py
    
    This will start a local server and open the app in your default web browser. You can enter a stock ticker (e.g., AAPL, TSLA, NVDA) and click the "Fetch Data" button to see the stock data, moving averages, and price predictions. 
    Conclusion 
    In this tutorial, we learned how to use the  Prophet  library to forecast time series data in Python. We covered the key concepts behind Prophet and demonstrated how to use it to predict stock prices. 
    Prophet is a powerful tool for time series forecasting and is particularly useful for business applications. It is easy to use and provides accurate predictions with minimal configuration. 
    If you're interested in learning more about time series forecasting, check out the following resources: 
    
    How to Use ARIMA for Time Series Forecasting in Python
    How to Use LSTM Networks for Time Series Forecasting
    
    To learn more about the Prophet library, visit the official documentation: 
    
    Prophet Documentation
    
    If you have any questions or thoughts to share, feel free to reach out in the comments below! 
    About the authors 
    
    Ramesh Sannareddy 
    Ramesh is a seasoned Data Scientist and ML Engineer with 15 years of experience in the tech industry. He has worked for global tech companies and founded  Data Science Dojo, a data science training company. 
    
    Taurai Mutimutema 
    Taurai is a systems analyst with a knack for writing, which was probably sparked by the need to document technical processes during code and implementation sessions. He enjoys learning new technologies and solving programming problems. Taurai has a Bachelor of Science in Computer Science and is always looking for opportunities to put his skills into practice.'''