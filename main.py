import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Stock Price Predictor App", layout="wide")

# Title
st.title("Stockify")

# Stock input
stock = st.text_input("Enter the Stock ID (e.g., AAPL, GOOG)", "GOOG").upper()

# Date range input
start_date = st.date_input("Start Date", datetime(2003, 1, 1))
end_date = st.date_input("End Date", datetime.now())

# Fetch stock data
if st.button("Fetch Data"):
    try:
        reli_data = yf.download(stock, start=start_date, end=end_date)
        if reli_data.empty:
            st.error("No data found for the specified stock.")
        else:
            st.subheader("Stock Data")
            st.write(reli_data)

            # Load the pre-trained model
            model = load_model("Latest_stock_price_model.keras")

            # Prepare the data for predictions
            splitting_len = int(len(reli_data) * 0.7)
            x_test = pd.DataFrame(reli_data.Close[splitting_len:])

            # Calculate moving averages
            for ma in [100, 200, 250]:
                reli_data[f'MA_for_{ma}_days'] = reli_data['Close'].rolling(ma).mean()

            # Plot moving averages as candlestick chart
            st.subheader('Candlestick Chart with Moving Averages')
            fig = go.Figure()

            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=reli_data.index,
                open=reli_data['Open'],
                high=reli_data['High'],
                low=reli_data['Low'],
                close=reli_data['Close'],
                name='Candlestick'
            ))

            # Add moving averages
            for ma in [100, 200, 250]:
                fig.add_trace(go.Scatter(
                    x=reli_data.index,
                    y=reli_data[f'MA_for_{ma}_days'],
                    mode='lines',
                    name=f'MA for {ma} days'
                ))

            st.plotly_chart(fig)

            # Prepare data for model input
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(x_test[['Close']])
            x_data, y_data = [], []

            for i in range(100, len(scaled_data)):
                x_data.append(scaled_data[i-100:i])
                y_data.append(scaled_data[i])

            x_data, y_data = np.array(x_data), np.array(y_data)

            # Make predictions
            predictions = model.predict(x_data)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)

            # Prepare plotting data
            plotting_data = pd.DataFrame({
                'Original Test Data': inv_y_test.reshape(-1),
                'Predictions': inv_pre.reshape(-1)
            }, index=reli_data.index[splitting_len + 100:])

            # Plot original vs predicted values as a line chart
            st.subheader("Original vs Predicted Values")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=plotting_data.index,
                y=plotting_data['Original Test Data'],
                mode='lines',
                name='Original'
            ))
            fig2.add_trace(go.Scatter(
                x=plotting_data.index,
                y=plotting_data['Predictions'],
                mode='lines',
                name='Predicted'
            ))
            st.plotly_chart(fig2)

            # Final plot with all data
            st.subheader('Complete Data Visualization')
            fig3 = go.Figure()
            fig3.add_trace(go.Candlestick(
                x=reli_data.index,
                open=reli_data['Open'],
                high=reli_data['High'],
                low=reli_data['Low'],
                close=reli_data['Close'],
                name='Candlestick'
            ))
            fig3.add_trace(go.Scatter(
                x=plotting_data.index,
                y=plotting_data['Predictions'],
                mode='lines',
                name='Predicted Price'
            ))
            st.plotly_chart(fig3)

    except Exception as e:
        st.error(f"An error occurred: {e}")
