import pandas as pd
import yfinance as yf
import numpy as np


def get_data(ticker, start_date, end_date):    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def load_data():    
    data = pd.read_csv('./Files/oil_prices.csv', sep=';')
    data.columns = ['Date', 'Close']
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def load_gdp_data():
    gdp_data = pd.read_csv('./Files/world_gdp.csv', sep=';')
    gdp_data['Year'] = gdp_data['Year'].astype(int)
    gdp_data['Global GDP'] = gdp_data['Global GDP'].astype(float)
    return gdp_data


def load_sp500_data():
    data = pd.read_csv('./Files/sp500_data.csv')
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])   
    return data


def create_features(df):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['lag1'] = df['Close'].shift(1)
    df['lag2'] = df['Close'].shift(2)
    df['lag3'] = df['Close'].shift(3)
    df['lag7'] = df['Close'].shift(7)
    df.dropna(inplace=True)
    return df



def forecast_next_days(model, df, days=7):
    last_row = df.iloc[-1]
    forecast_dates = pd.date_range(start=last_row['Date'], periods=days+1)[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates})
    forecast_df['year'] = forecast_df['Date'].dt.year
    forecast_df['month'] = forecast_df['Date'].dt.month
    forecast_df['day'] = forecast_df['Date'].dt.day
    forecast_df['dayofweek'] = forecast_df['Date'].dt.dayofweek

    forecasts = []
    last_values = last_row[['Close', 'lag1', 'lag2', 'lag3', 'lag7']].values.flatten()

    for i in range(days):
        features = np.array([
            forecast_df.loc[i, 'year'],
            forecast_df.loc[i, 'month'],
            forecast_df.loc[i, 'day'],
            forecast_df.loc[i, 'dayofweek'],
            last_values[0],
            last_values[1],
            last_values[2], 
            last_values[3] 
        ]).reshape(1, -1)
        
        yhat = model.predict(features)[0]
        forecasts.append(yhat)
        last_values = np.roll(last_values, 1)
        last_values[0] = yhat
   
    forecast_df['Close'] = forecasts
    return forecast_df