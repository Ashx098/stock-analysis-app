import streamlit as st
import yfinance as yf

# Title
st.title("📈 Stock Analysis & Prediction System")

# User input
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "NVDA")

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch last 5 years of data
    return df

# Button to fetch data
if st.button("Fetch Data"):
    stock_data = get_stock_data(stock_ticker)
    
    # Show data if fetched successfully
    if not stock_data.empty:
        st.subheader(f"Stock Data for {stock_ticker.upper()}")
        st.write(stock_data.head())  # Display first few rows
    else:
        st.error("Failed to retrieve stock data. Please check the ticker.")

st.write("👆 Enter a stock ticker and click 'Fetch Data' to get started!")
