import yfinance as yf 
import pandas as pd
def get_data(name):
    data = yf.download(name)
    df =pd.DataFrame(data)
    new={}
    new["Date"]=df.index.values
    print(new["Date"].shape)
    new["Close"]=df["Close"].values.reshape(len(df["Close"].values))
    new["High"]=df["High"].values.reshape(len(df["High"].values))
    new["Low"]=df["Low"].values.reshape(len(df["Low"].values))
    new["Open"]=df["Open"].values.reshape(len(df["Open"].values))
    new["Volume"]=df["Volume"].values.reshape(len(df["Volume"].values))
    new=pd.DataFrame(new)
    return new