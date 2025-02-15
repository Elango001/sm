# Stock Price Prediction Web App

## Overview
This project is a **Stock Price Prediction Web App** built using **Streamlit**. The application allows users to select a stock from a predefined list, view historical stock data, and visualize actual vs. predicted prices. The model uses **machine learning** techniques to predict future stock prices and provides technical indicators like SMA, EMA, and Bollinger Bands.

## Features
- **Stock Selection**: Choose from a list of stocks like AAPL, GOOGL, AMZN, TSLA, etc.
- **Data Visualization**: Displays historical stock prices.
- **Stock Price Prediction**: Uses an LSTM model to predict future stock prices.
- **Performance Metrics**: Displays RMSE and R² scores to evaluate model accuracy.
- **Technical Indicators**:
  - **Simple Moving Average (SMA)**
  - **Exponential Moving Average (EMA)**
  - **Bollinger Bands**

## Disclaimer
> **This stock prediction model is developed for educational purposes only.**
> Predictions generated by this model are not financial advice. Stock markets are influenced by many factors, and no model can guarantee future performance.

## Installation
To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Elango001/sm/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py
   ```

## Usage
1. Select a stock from the dropdown menu.
2. View the loaded historical data.
3. See actual vs. predicted stock prices.
4. Analyze model performance (RMSE & R² score).
5. View stock indicators (SMA, EMA, Bollinger Bands).
6. Predict future stock prices for the next 10 days.

## Model Details
- The application loads stock data using `MODEL.py`.
- Data is preprocessed by extracting features and normalizing values.
- A machine learning model is trained using **LSTM** (Long Short-Term Memory) networks.
- The model is evaluated using RMSE and R² scores.
- Predictions for future prices are displayed graphically.

## Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## Contact
For any questions or feedback, feel free to reach out!

- **Email**: elangos195@gmail.com
- **GitHub**: https://github.com/Elango001

---

Enjoy predicting stock prices! 🚀

