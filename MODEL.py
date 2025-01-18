import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout,Input
from keras.losses import Huber
from keras.optimizers import Adam, RMSprop
import optuna
import tensorflow as tf
import joblib
from DB import Find_File
np.random.seed(42)
tf.random.set_seed(42)
def data(name):
    file=Find_File(name)
    file=file[-(10*365):]
    return file
#file=data("AAPL")
def imp():
    import joblib
    predictor=["Close","MACD","EMA12","EMA26","MA5","Bollinger_Upper","Bollinger_Lower"]
    target="Close"
    scalar_data=joblib.load("scalar_data")
    scalar_target=joblib.load("scalar_target")
    return predictor,target,scalar_data,scalar_target
def features(file,window=20, num_std=2):
    rolling_mean = file['Close'].rolling(window=window).mean()
    rolling_std = file['Close'].rolling(window=window).std()
    file['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    file['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    file["EMA12"] = file["Close"].ewm(span=12, adjust=False).mean()
    file["EMA26"] = file["Close"].ewm(span=26, adjust=False).mean()
    file["MA5"]=file["Close"].rolling(window=5).mean()
    file['Bollinger_Upper'] = file['Bollinger_Upper'].fillna(file['Bollinger_Upper'].mean())
    file['Bollinger_Lower'] = file['Bollinger_Lower'].fillna(file['Bollinger_Lower'].mean())
    file['MA5'] = file['MA5'].fillna(file['MA5'].mean())
    file["MACD"] = file["EMA12"] - file["EMA26"]
    file=file.dropna()
    return file
def data_split(file,predictor,target,scalar_data,scalar_target):
    import numpy as np
    from sklearn.model_selection import train_test_split
    train,test = train_test_split(file, test_size=0.2, shuffle=False)    
    scaled_train=scalar_data.fit_transform(train[predictor])
    scaled_train_target=scalar_target.fit_transform(train[[target]])
    scaled_test=scalar_data.transform(test[predictor])
    scaled_test_target=scalar_target.transform(test[[target]])
    def create_dataset(data,close,length=5):
        X, y = [], []
        for i in range(len(data) - length):
            X.append(data[i:i+length])
            y.append(close[i+length])
        return np.array(X),np.array(y)
    Xtrain,ytrain=create_dataset(scaled_train,scaled_train_target)
    Xtest,ytest=create_dataset(scaled_test,scaled_test_target)
    return Xtrain,ytrain,Xtest,ytest
def objective(trial):
    global Xtrain,ytrain,Xtest,ytest
    units = trial.suggest_int('units', 50, 150)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2,log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    model = Sequential()
    model.add(Input(shape=(Xtrain.shape[1],Xtrain.shape[2])))
    model.add(LSTM(units, return_sequences=False,activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(1,activation="linear")) 
    model.compile(optimizer=optimizer, loss=Huber(), metrics=["mse"])
    history = model.fit(
        Xtrain,ytrain,
        validation_data=(Xtest,ytest),
        epochs=20, 
        batch_size=32,
        verbose=0  
    )
    val_loss, val_mse = model.evaluate(Xtest,ytest, verbose=0)
    return val_loss 
def model_val():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params
def overfitting():
    from keras.callbacks import EarlyStopping,ReduceLROnPlateau
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,         
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  
        factor=0.5,          
        patience=5,          
        min_lr=1e-6         
   )
    return early_stopping,reduce_lr
def model(Xtrain,ytrain,Xtest,ytest):
    #tweaks=model_val()
    tweaks= {'units': 128, 'dropout': 0.3189847791811019, 'learning_rate': 0.0006148418365494306, 'optimizer': 'adam'}
    """final_model = Sequential()
    final_model.add(Input(shape=(Xtrain.shape[1],Xtrain.shape[2])))
    final_model.add(LSTM(tweaks["units"], return_sequences=False,activation="relu"))
    final_model.add(Dropout(tweaks["dropout"]))    
    final_model.add(Dense(24, activation="relu"))
    final_model.add(Dense(1,activation="linear"))"""
    final_model=load_model("model.keras")
    early_stopping,reduce_lr=overfitting()
    if tweaks["optimizer"]=="adam":
        final_model.compile(optimizer=Adam(learning_rate=tweaks["learning_rate"]), loss=Huber(), metrics=["mse"])
    else:
        final_model.compile(optimizer=RMSprop(learning_rate=tweaks["learning_rate"]), loss=Huber(), metrics=["mse"])
    history = final_model.fit(
        Xtrain,ytrain,
        validation_data=(Xtest,ytest),
        epochs=10,  # You can increase epochs for final training
        batch_size=32,
        callbacks=[early_stopping,reduce_lr],
        verbose=1
        )
    return final_model
def model_performance(model,Xtest,ytest,scalar_target):
    preds=model.predict(Xtest)
    actual_preds=scalar_target.inverse_transform(preds)
    actual_prices=scalar_target.inverse_transform(ytest)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(actual_prices,actual_preds))
    mae = mean_absolute_error(actual_prices,actual_preds)
    r2 = r2_score(actual_prices,actual_preds)
    return {"RMSE":rmse,"MAE":mae,"R2":r2}
def prediction(model,file,scalar_target,scalar_data):
    scaled=scalar_data.transform(file)
    def dataset(data, length=5):
        X= []
        for i in range(len(data) - length):
            X.append(data[i:i+length])
        return np.array(X)
    X=dataset(scaled[-6:])
    pred=model.predict(X)
    last_day_data={}
    last_day_data['Close']= scalar_target.inverse_transform(pred)[0][0]
    file.loc[len(file)]=last_day_data
    rolling_mean = file['Close'].rolling(window=20).mean()
    rolling_std = file['Close'].rolling(window=20).std()
    file['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    file['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
    file["EMA12"] =file["Close"].ewm(span=12, adjust=False).mean()
    file["EMA26"] = file["Close"].ewm(span=26, adjust=False).mean()
    file["MA5"] =file["Close"].rolling(window=5).mean()
    file['Bollinger_Upper'] = file['Bollinger_Upper'].fillna(file['Bollinger_Upper'].mean())
    file['Bollinger_Lower'] = file['Bollinger_Lower'].fillna(file['Bollinger_Lower'].mean())
    file['MA5'] = file['MA5'].fillna(file['MA5'].mean())
    file["MACD"] =file["EMA12"] - file["EMA26"]
    return file
def model_pred(model,file,scalar_data,scalar_target,num_days=10):
    for i in range(num_days):
        file=prediction(model,file,scalar_target,scalar_data)   
    return pd.DataFrame(file["Close"][-(num_days):])
#file=features(file,predictor)
#Xtrain,ytrain,Xtest,ytest=data_split(file,predictor,target,scalar_data,scalar_target)
#model=model()
#pref=model_performance(model)
#preds=model_pred(model,file)
#print(preds)