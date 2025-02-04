import streamlit as st  # Import Streamlit for web app interface
import matplotlib.pyplot as plt  # Import Matplotlib for plotting graphs
import MODEL as md  # Import custom module 'MODEL' containing functions for stock prediction
import pandas as pd

# Display a disclaimer about the stock prediction model
st.markdown(
    """
    ### Disclaimer:
    This stock prediction model is developed for **educational purposes only**. 
    The predictions generated by this model are not financial advice and should not be used for making investment decisions. 

    Stock markets are influenced by a wide range of factors, and no model can guarantee future performance. 
    Use this model at your own risk, and always consult with a professional financial advisor before making any investment decisions.
    """,
)

# Dropdown for selecting a stock from a predefined list
stocks = ['Select a stock', 'AAPL', 'ADBE', 'GOOGL', 'AMZN', 'AVGO', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
name = st.selectbox("Select a stock:", stocks)

# If no stock is selected, display a welcome message
if name == "Select a stock":
    st.markdown("""**Welcome**""")
else:
    def load_data(stock_name):
        return md.data(stock_name)  # Fetch stock data using function from 'MODEL'

    file = load_data(name)  # Load stock data
    st.write("Loaded Data:", file)  # Display the loaded data

    final_file = file[-10:]
    file = file[:-10]

    # Load necessary preprocessing functions from MODEL
    predictor, target, scalar_data, scalar_target = md.imp()

    # Process data and extract features
    file = md.features(file)

    # Split the data into training and testing sets
    Xtrain, ytrain, Xtest, ytest = md.data_split(file, predictor, target, scalar_data, scalar_target)

    @st.cache_resource  # Cache the trained model
    def train_model(Xtrain, ytrain, Xtest, ytest):
        return md.model(Xtrain, ytrain, Xtest, ytest)  # Train and return the model

    model = train_model(Xtrain, ytrain, Xtest, ytest)  # Train the model

    # Function to plot actual vs predicted stock prices
    def plot_predictions_with_dates(file, Xtrain, Xtest, ytrain, ytest, scalar_target):
        actual = scalar_target.inverse_transform(ytrain)  # Convert scaled values to actual prices
        pred_train = scalar_target.inverse_transform(model.predict(Xtrain))  # Predict and scale back training data

        dates = file["Date"][:len(actual)]  # Get corresponding dates for training data

        # Plot actual vs predicted training data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, actual, label="Actual Train")
        ax.plot(dates, pred_train, label="Predicted Train")

        # Predict on test data
        dates = file["Date"][-len(ytest):]
        actual_test = scalar_target.inverse_transform(ytest)
        pred_test = scalar_target.inverse_transform(model.predict(Xtest))

        ax.plot(dates, actual_test, label="Actual Test")
        ax.plot(dates, pred_test, label="Predicted Test")

        ax.set_title(f"Stock Predictions for {name}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price", fontsize=12)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        ax.legend()
        return fig

    fig = plot_predictions_with_dates(file, Xtrain, Xtest, ytrain, ytest, scalar_target)
    st.pyplot(fig)  # Display the plot in the Streamlit app

    # Evaluate the model's performance
    perf = md.model_performance(model, Xtest, ytest, scalar_target)
    st.write("Prediction Error (RMSE):", perf["RMSE"])  # Display RMSE
    st.write("Pattern Explained (R²):", perf["R2"])  # Display R² score

    # Function to predict future stock prices
    def pred_or(model, file, scalar_data, scalar_target, num_days=10):
        return md.model_pred(model, file[predictor], scalar_data, scalar_target, num_days)

    predicted = pred_or(model, file, scalar_data, scalar_target, num_days=10)

    from datetime import datetime, timedelta

    def get_next_10_dates(start_date, date_format="%Y-%m-%d"):
        start = datetime.strptime(start_date, date_format)
        next_dates = [(start + timedelta(days=i)).strftime(date_format) for i in range(1, 11)]
        return next_dates

    # Function to plot future predictions
    def pred_plot(predicted, final_file, date, num_days=10):
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = get_next_10_dates(date)
        ax.plot(dates, predicted, marker="o", label=f"Predicted Next {num_days} Days")
        ax.plot(dates, final_file["Close"], marker="*", label="Actual")
        ax.legend()
        return fig

    date = file["Date"].iloc[-1].strftime("%Y-%m-%d")
    new_fig = pred_plot(predicted, final_file, date)
    st.pyplot(new_fig)  # Display future predictions plot

    # Function to plot stock indicators (SMA, EMA, Bollinger Bands)
    def plot_features(file, data, days):
        fig, ax = plt.subplots(figsize=(12, 6))

        if data == "SMA":  # Simple Moving Average
            ax.plot(file["Date"], file["Close"], label="Stock Data")
            file[f"SMA{days}"] = file["Close"].rolling(window=days).mean()  # Calculate SMA
            ax.plot(file["Date"], file[f"SMA{days}"], label=f"SMA{days}")

        elif data == "EMA":  # Exponential Moving Average
            ax.plot(file["Date"], file["Close"], label="Stock Data")
            file[f"EMA{days}"] = file["Close"].ewm(span=days, adjust=False).mean()  # Calculate EMA
            ax.plot(file["Date"], file[f"EMA{days}"], label=f"EMA{days}")

        elif data == "Bollinger Up and Bollinger Down":  # Bollinger Bands
            ax.plot(file["Date"], file["Close"], label="Stock Data")
            rolling_mean = file['Close'].rolling(window=20).mean()
            rolling_std = file['Close'].rolling(window=20).std()
            file['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)  # Upper Bollinger Band
            file['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)  # Lower Bollinger Band
            ax.plot(file["Date"], file["Bollinger_Upper"], label="Bollinger Upper")
            ax.plot(file["Date"], file["Bollinger_Lower"], label="Bollinger Lower")

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price", fontsize=12)
        plt.xticks(rotation=45)
        ax.legend()
        return fig

    # Streamlit widgets for selecting technical indicators
    data = st.selectbox(label="Select an option", options=["SMA", "EMA", "Bollinger Up and Bollinger Down"])
    days = st.slider("Select the number of days", min_value=5, max_value=200, step=5, value=5)

    # Plot selected indicator and display it in the app
    fig = plot_features(file, data, days)
    st.pyplot(fig)
