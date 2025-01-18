import streamlit as st
import matplotlib.pyplot as plt
import MODEL as md

# Stock Selection
stocks = ['None','AAPL','ADBE','GOOGL','AMZN','AVGO','GOOGL','META','MSFT','NVDA','TSLAS']
name = st.selectbox("Select a stock:", stocks)
if name=="None":
    st.write("welcome")
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
        dates=file.index[:len(actual)]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, actual )
        ax.plot(dates, pred_train )
        dates=file.index[-len(ytest):]
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
    st.write("Model Performance:", perf)

    def pred_or(model,file,scalar_data,scalar_target,num_days=10):
        return md.model_pred(model,file[predictor],scalar_data,scalar_target,num_days)
    predicted=pred_or(model,file,scalar_data,scalar_target,num_days=10)
    def pred_plot(predicted,scalar_target,num_days=10):
        fig,ax=plt.subplots(figsize=(12,6))
        ax.plot(predicted)
        ax.legend([f"predicted next {num_days}"])
        return fig
    new_fig=pred_plot(predicted,scalar_target)
    st.pyplot(new_fig)

