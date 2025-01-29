import streamlit as st
import matplotlib.pyplot as plt
import MODEL as md
st.markdown(
    """
    ### Disclaimer:
    This stock prediction model is developed for **educational purposes only**. 
    The predictions generated by this model are not financial advice and should not be used for making investment decisions. 

    Stock markets are influenced by a wide range of factors, and no model can guarantee future performance. 
    Use this model at your own risk, and always consult with a professional financial advisor before making any investment decisions.
    """,
    unsafe_allow_html=True,
)
# Stock Selection
stocks = ['Select a stock','AAPL','ADBE','GOOGL','AMZN','AVGO','GOOG','META','MSFT','NVDA','TSLA']
name = st.selectbox("Select a stock:", stocks)
if name=="Select a stock":
    st.markdown("""###**welcome**""")
# Cached Data Loading
else:
    @st.cache_data
    def load_data(stock_name):
        return md.data(stock_name)
    file = load_data(name)
    st.write("Loaded Data:", file)
    predictor, target, scalar_data, scalar_target = md.imp()
    file = md.features(file)
    Xtrain, ytrain, Xtest, ytest = md.data_split(file, predictor, target, scalar_data, scalar_target)
    @st.cache_resource
    def train_model(Xtrain, ytrain, Xtest, ytest):
        return md.model(Xtrain, ytrain, Xtest, ytest)
    model =train_model(Xtrain, ytrain, Xtest, ytest)
    def plot_predictions_with_dates(file, Xtrain,Xtest ,ytrain,ytest, scalar_target):
        actual = scalar_target.inverse_transform(ytrain)
        pred = model.predict(Xtrain)
        pred_train = scalar_target.inverse_transform(pred)
        dates=file["Date"][:len(actual)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, actual )
        ax.plot(dates, pred_train )
        dates=file["Date"][-len(ytest):]
        actual_test=scalar_target.inverse_transform(ytest)
        pred_test=model.predict(Xtest)
        pred_test=scalar_target.inverse_transform(pred_test)
        ax.plot(dates,actual_test)
        ax.plot(dates,pred_test)
        ax.legend(["actual_train","predict_train","actual_test","predict_test"])
        ax.set_title(f"Stock Predictions for {name}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Stock Price", fontsize=12)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        return fig
    fig = plot_predictions_with_dates(file, Xtrain,Xtest, ytrain,ytest, scalar_target)
    st.pyplot(fig)

    perf = md.model_performance(model, Xtest, ytest, scalar_target)
    st.write("Prediction Error:",perf["RMSE"])
    st.write("Pattern Explained:",perf["R2"])
    def pred_or(model,file,scalar_data,scalar_target,num_days=10):
        return md.model_pred(model,file[predictor],scalar_data,scalar_target,num_days)
    predicted=pred_or(model,file,scalar_data,scalar_target,num_days=10)
    def pred_plot(predicted,num_days=10):
        fig,ax=plt.subplots(figsize=(12,6))
        ax.plot(predicted)
        ax.legend([f"predicted next {num_days}"])
        return fig
    new_fig=pred_plot(predicted)
    st.pyplot(new_fig)
    def plot_features(file,data,days):
        fig,ax=plt.subplots(figsize=(12,6))
        if data=="SMA":
            ax.plot(file["Date"],file["Close"])
            file[f"SMA{days}"]=file["Close"].rolling(window=days).mean()
            ax.plot(file["Date"],file[f"SMA{days}"])
            ax.legend([f"Stock Data","SMA{days}"])
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            plt.xticks(rotation=45)
            return fig
        if data=="EMA":
            ax.plot(file["Date"],file["Close"])
            file[f"EMA{days}"]=file["Close"].ewm(span=days,adjust=False).mean()
            ax.plot(file["Date"],file[f"EMA{days}"])
            ax.legend([f"Stock Data","EMA{days}"])
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            plt.xticks(rotation=45)
            return fig
        if data=="Bollinger Up and Bollinger Down":
            ax.plot(file["Date"],file["Close"])
            rolling_mean = file['Close'].rolling(window=20).mean()
            rolling_std = file['Close'].rolling(window=20).std()
            file['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
            file['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
            ax.plot(file["Date"],file["Bollinger_Upper"])
            ax.plot(file["Date"],file["Bollinger_Lower"])
            ax.legend(["Stock Data","Bollinger_upper","Bollinger_lower"])
            #plt.fill_between(file['Date'], file['Bollinger_Upper'], file['Bollinger_Lower'], color='gray')
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            plt.xticks(rotation=45)
            return fig    
    data=st.selectbox(label="Select an option",options=["SMA","EMA","Bollinger Up and Bollinger Down"])
    days=st.slider("Select the number of days",min_value=5,max_value=200,step=5,value=5)
    fig=plot_features(file,data,days)
    st.pyplot(fig)

