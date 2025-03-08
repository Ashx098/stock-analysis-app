import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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
        st.write(stock_data.head())  # Show first few rows
        
        # Plot Closing Price
        st.subheader("Stock Closing Price Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data["Close"], label="Closing Price", color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price (USD)")
        ax.set_title(f"{stock_ticker.upper()} Closing Price History")
        ax.legend()
        st.pyplot(fig)  # Display plot in Streamlit

    else:
        st.error("Failed to retrieve stock data. Please check the ticker.")

st.write("👆 Enter a stock ticker and click 'Fetch Data' to get started!")
