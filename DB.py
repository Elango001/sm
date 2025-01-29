import yfinance as yf  # Import Yahoo Finance library to fetch stock data
import pandas as pd  # Import pandas for data manipulation

def get_data(name):
    """
    Fetch historical stock data for the given ticker symbol and return a formatted DataFrame.
    
    Args:
        name (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
    
    Returns:
        pd.DataFrame: A DataFrame containing Date, Close, High, Low, Open, and Volume columns.
    """
    data = yf.download(name)  # Download stock data using Yahoo Finance
    df = pd.DataFrame(data)  # Convert downloaded data to a DataFrame
    
    new = {}  # Create an empty dictionary to store reformatted data
    new["Date"] = df.index.values  # Extract dates from the index
    # Reshape and store individual columns into the dictionary
    new["Close"] = df["Close"].values.reshape(len(df["Close"].values))  
    new["High"] = df["High"].values.reshape(len(df["High"].values))  
    new["Low"] = df["Low"].values.reshape(len(df["Low"].values))  
    new["Open"] = df["Open"].values.reshape(len(df["Open"].values))  
    new["Volume"] = df["Volume"].values.reshape(len(df["Volume"].values))  
    
    new = pd.DataFrame(new)  # Convert the dictionary back into a DataFrame
    return new  # Return the formatted DataFrame
