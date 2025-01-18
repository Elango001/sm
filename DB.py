import yfinance as yf
from datetime import datetime,timezone
import pandas as pd
from dotenv import load_dotenv
from os import getenv
load_dotenv()
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
def Find_File(name:str):
    client=MongoClient(getenv("DATABASEURL"),connectTimeoutMS=30000,socketTimeoutMS=30000,
    server_api=ServerApi('1'))
    db=client["File_Store"]
    collection=db[name]
    file=collection.find()
    df = pd.DataFrame(list(file))
    df = df.drop(columns=['_id'])
    df=df.set_index("Date")
    return df
def get_data(name):
    client = MongoClient(getenv("DATABASEURL"), connectTimeoutMS=1200000, socketTimeoutMS=1200000, server_api=ServerApi('1'))
    db = client["File_Store"]
    collection = db[name]
    data = yf.download(name, interval="1d")
    df =pd.DataFrame(data)
    yo=df[-1:].values.reshape(5)
    Date=df[-1:].index.values[0]
    Date = datetime.fromtimestamp(Date.astype('datetime64[s]').astype('int'), tz=timezone.utc)
    new={}
    new["Date"]=Date
    new["Close"]=yo[0]
    new["High"]=yo[1]
    new["Low"]=yo[2]
    new["Open"]=yo[3]
    new["Volume"]=yo[4]
    t=collection.find_one({"Date":new["Date"]})
    if t==None:
        collection.insert_one(new)
    else:
        pass