import base64
from urllib.error import URLError
import pandas as pd
import numpy as np
import streamlit as st
# import yfinance as yf
# !pip install yahoo_fin
from yahoo_fin.stock_info import get_data
from st_on_hover_tabs import on_hover_tabs
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def get_stock_data(stock):
    df = get_data(stock, start_date = None, end_date = None, index_as_date = False, interval ='1d')
    return df

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


try:
    st.set_page_config(
        page_title="📈 Stock Price Prediction 📉",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Stock Price Prediction using Machine Learning"
        }
    )

    # add_bg_from_local('bg.jpeg')
    st.markdown('<style>' + open('./style.css').read() +'</>', unsafe_allow_html=True)
    with st.sidebar:
        
        tabs = on_hover_tabs(tabName=['Home','Trends','Model','Accuracy'],iconName=['home','economy','dashboard', 'speed'], default_choice=0)

    if tabs == 'Model':
        st.title("Stock Price Prediction 📈📉")
        # st.write('Hello :sunglasses:')
        stock = st.selectbox('Select a Stock: ', ("APOLLOHOSP.NS", "TATACONSUM.NS", "TATASTEEL.NS", "RELIANCE.NS", "LT.NS", "BAJAJ-AUTO.NS", "WIPRO.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS",
                                                  "ULTRACEMCO.NS", "BRITANNIA.NS", "TITAN.NS", "INDUSINDBK.NS", "ICICIBANK.NS", "ONGC.NS", "NTPC.NS", "ITC.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
                                                  "TECHM.NS", "HDFCLIFE.NS", "HINDALCO.NS", "BHARTIARTL.NS", "CIPLA.NS", "TCS.NS", "ADANIENT.NS", "HEROMOTOCO.NS", "MARUTI.NS", "COALINDIA.NS",
                                                  "BPCL.NS", "HCLTECH.NS", "ADANIPORTS.NS", "DRREDDY.NS", "EICHERMOT.NS", "ASIANPAINT.NS", "GRASIM.NS", "JSWSTEEL.NS", "DIVISLAB.NS", "TATACONSUM.NS",
                                                  "SBIN.NS", "HDFCBANK.NS", "HDFC.NS", "WIPRO.NS", "UPL.NS", "POWERGRID.NS", "TATAPOWER.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "HINDUNILVR.NS",
                                                  "SBILIFE.NS", "INFY.NS", "AXISBANK.NS"))
        # st.markdown(
        #     "<p style='color: black;'>You selected: ", unsafe_allow_html=True)
        st.write("You selected: ",stock)
        df = get_stock_data(stock)
        if not stock:
            st.error("Please select at least one country.")
        else:
            if st.button('Show Data'):
                st.dataframe(df, use_container_width=True)
            # else:
            #     st.write('Continue')
            # tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])
            mean = df['open'].mean()
            df['open'] = df['open'].fillna(mean)

            mean = df['high'].mean()
            df['high'] = df['high'].fillna(mean)

            mean = df['low'].mean()
            df['low'] = df['low'].fillna(mean)

            mean = df['close'].mean()
            df['close'] = df['close'].fillna(mean)

            X = df[['open', 'high', 'low']]
            y = df['close'].values.reshape(-1, 1)

            split_size = st.select_slider(
                'Select ratio for train test splitting: ',
                options=[0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5])
            st.write('Training data: ', (1-split_size)*100,'%')
            st.write('Testing data: ', split_size*100,'%')

            #Splitting our dataset to Training and Testing dataset
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_size, random_state=42)

            #Fitting Linear Regression to the training set
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(X_train, y_train)

            #predicting the Test set result
            y_pred = reg.predict(X_test)
            o = df['open'].values
            h = df['high'].values
            l = df['low'].values

            tab1, tab2 = st.tabs(
                ["Custom Input", "Present Day"])

            with tab2:
                n = len(df)
                pred = []
                for i in range(0, n):
                    open = o[i]
                    high = h[i]
                    low = l[i]
                    output = reg.predict([[open, high, low]])
                    pred.append(output)

                pred1 = np.concatenate(pred)
                predicted = pred1.flatten().tolist()
                df['Prediction'] = predicted
                t = predicted[-1]
                with st.expander("Predict"):
                    st.subheader("Predicted Closing Price: ")
                    st.subheader(t)
                    st.snow()

                with tab1:
                    st.subheader("Custom Input")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        open = st.number_input("Opening Price")

                    with col2:
                        high = st.number_input("Day High")

                    with col3:
                        low = st.number_input("Day Low")

                    output = reg.predict([[open, high, low]])
                    with st.expander("Predict"):
                        st.subheader("Predicted Closing Price: ")
                        st.subheader(output[0][0])
                        st.balloons()
        
            st.markdown(
                "<h3 style='color: black;text-align: center;'><b>Comparision between Actual vs. Predicted </b></h3>", unsafe_allow_html=True)
            # fig = st.line_chart(data=df, x='date', y='close')
            
            chart_data = pd.DataFrame(
                df, columns=['close','Prediction'])

            st.line_chart(chart_data)

    elif tabs == 'Home':
        st.title("Home")
        st.write("<b>Stock price prediction is the task of forecasting the future value of a stock based on historical data and other relevant factors. Machine learning has become an increasingly popular approach to predicting stock prices, as it can analyze large amounts of data and identify complex patterns that may not be immediately apparent to humans.</b>",unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(" ")

        with col2:
            st.image("media/wordcloud.png", width=600)
        with col3:
            st.write(" ")

        with col4:
            st.write(" ")
        
        with col5:
            st.write(" ")

        st.write("<b>There are several machine learning techniques that can be used for stock price prediction, including regression analysis, time series analysis, and neural networks. Regression analysis involves fitting a mathematical model to historical data and using it to make predictions about future stock prices.</b>", unsafe_allow_html=True)
        st.write(" ")
        
        st.write("<b>Predicting the closing price of a particular stock listed on the National Stock Exchange (NSE) is a common application of machine learning in finance. The closing price is the final price at which a stock is traded on a given day, and predicting it accurately can help investors make informed trading decisions.</b>", unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        col1, col2, col3, col4,col5 = st.columns(5)
        with col1:
            st.image("media/icici.png", width=140)
            st.image("media/sbi.png", width=170)

        with col2:
            st.image("media/RIL.png",width=170)
            st.image("media/eicher.png", width=180)
            st.image("media/brit.png",width=180)
            

        with col3:
            st.image("media/tatapower.png", width=170)
            st.write("\n")
            st.image("media/mrf.png", width=170)
            st.write("\n")
            st.image("media/unilever.png", width=150)
            
            

        with col4:
            st.image("media/hdfc.jpg", width=130)
            st.image("media/nestle.jpg", width=170)
            st.write("\n")
            
            st.write("\n")
            st.image("media/bajaj.png", width=170)
     
        with col5:
            st.image("media/adani.png", width=160)
            st.image("media/titan.png", width=170)


    elif tabs == 'Accuracy':
        st.title("⏳Evaluation Metrics⌛⏱️")
        stock = st.selectbox('Select a Stock: ', ("APOLLOHOSP.NS", "TATACONSUM.NS", "TATASTEEL.NS", "RELIANCE.NS", "LT.NS", "BAJAJ-AUTO.NS", "WIPRO.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS",
                                                  "ULTRACEMCO.NS", "BRITANNIA.NS", "TITAN.NS", "INDUSINDBK.NS", "ICICIBANK.NS", "ONGC.NS", "NTPC.NS", "ITC.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
                                                  "TECHM.NS", "HDFCLIFE.NS", "HINDALCO.NS", "BHARTIARTL.NS", "CIPLA.NS", "TCS.NS", "ADANIENT.NS", "HEROMOTOCO.NS", "MARUTI.NS", "COALINDIA.NS",
                                                  "BPCL.NS", "HCLTECH.NS", "ADANIPORTS.NS", "DRREDDY.NS", "EICHERMOT.NS", "ASIANPAINT.NS", "GRASIM.NS", "JSWSTEEL.NS", "DIVISLAB.NS", "TATACONSUM.NS",
                                                  "SBIN.NS", "HDFCBANK.NS", "HDFC.NS", "WIPRO.NS", "UPL.NS", "POWERGRID.NS", "TATAPOWER.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "HINDUNILVR.NS",
                                                  "SBILIFE.NS", "INFY.NS", "AXISBANK.NS"))

        st.write("You selected: ", stock)
        df = get_stock_data(stock)
        mean = df['open'].mean()
        df['open'] = df['open'].fillna(mean)

        mean = df['high'].mean()
        df['high'] = df['high'].fillna(mean)

        mean = df['low'].mean()
        df['low'] = df['low'].fillna(mean)

        mean = df['close'].mean()
        df['close'] = df['close'].fillna(mean)

        X = df[['open', 'high', 'low']]
        y = df['close'].values.reshape(-1, 1)

        split_size = st.select_slider(
            'Select ratio for train test splitting',
            options=[0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5])
        st.write('Training data: ', (1-split_size)*100, '%')
        st.write('Testing data: ', split_size*100, '%')

        #Splitting our dataset to Training and Testing dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_size, random_state=42)

        #Fitting Linear Regression to the training set
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        #predicting the Test set result
        y_pred = reg.predict(X_test)

        #Evaluating the model
        import sklearn.metrics as metrics
        r2 = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        
        col1, col2 = st.columns(2)
        
        col1.metric("R2 Score", r2, "±5%")
        col2.metric("Mean Absolute Error ", mae, "± 5%")
        col1.metric("Mean Squared Error", mse, "± 5%")
        col2.metric("Root Mean Squared Error", rmse, "± 5%")

    if tabs == 'Trends':
        st.header("Trend Analysis using Visualizations")
        st.markdown(
            "<hr></hr>", unsafe_allow_html=True)
        stock = st.selectbox('Select a Stock: ', ("APOLLOHOSP.NS", "TATACONSUM.NS", "TATASTEEL.NS", "RELIANCE.NS", "LT.NS", "BAJAJ-AUTO.NS", "WIPRO.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS",
                                                  "ULTRACEMCO.NS", "BRITANNIA.NS", "TITAN.NS", "INDUSINDBK.NS", "ICICIBANK.NS", "ONGC.NS", "NTPC.NS", "ITC.NS", "BAJFINANCE.NS", "NESTLEIND.NS",
                                                  "TECHM.NS", "HDFCLIFE.NS", "HINDALCO.NS", "BHARTIARTL.NS", "CIPLA.NS", "TCS.NS", "ADANIENT.NS", "HEROMOTOCO.NS", "MARUTI.NS", "COALINDIA.NS",
                                                  "BPCL.NS", "HCLTECH.NS", "ADANIPORTS.NS", "DRREDDY.NS", "EICHERMOT.NS", "ASIANPAINT.NS", "GRASIM.NS", "JSWSTEEL.NS", "DIVISLAB.NS", "TATACONSUM.NS",
                                                  "SBIN.NS", "HDFCBANK.NS", "HDFC.NS", "WIPRO.NS", "UPL.NS", "POWERGRID.NS", "TATAPOWER.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "HINDUNILVR.NS",
                                                  "SBILIFE.NS", "INFY.NS", "AXISBANK.NS"))
        # st.markdown(
        #     "<p style='color: black;'>You selected: ", unsafe_allow_html=True)
        
        st.write("You selected: ", stock)
        df = get_stock_data(stock)
    
        st.markdown(
             "<h5 style='color: black;text-align: center;'><b>Opening Price Trend </b></h5>", unsafe_allow_html=True)
        st.line_chart(data=df, x='date', y='open')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h5 style='color: black;text-align: center;'><b>Day High Trend </b></h5>", unsafe_allow_html=True)
            st.line_chart(data=df, x='date', y='high')

        with col2:
            st.markdown(
                "<h5 style='color: black;text-align: center;'><b>Day Low Trend </b></h5>", unsafe_allow_html=True)
            st.line_chart(data=df, x='date', y='low')
    
    
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )


