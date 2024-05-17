import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Carregar o modelo treinado
model = joblib.load('../Files/xgboost_model.pkl')

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
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
            last_values[0],  # lag1
            last_values[1],  # lag2
            last_values[2],  # lag3
            last_values[3]   # lag7
        ]).reshape(1, -1)
        
        yhat = model.predict(features)[0]
        forecasts.append(yhat)
        
        # Atualizar lag values
        last_values = np.roll(last_values, 1)
        last_values[0] = yhat
   
    forecast_df['Close'] = forecasts
    return forecast_df

st.title('Previsão de Preços do Petróleo Brent')

st.write("""
### Dados Históricos
""")

ticker = 'BZ=F'
start_date = st.date_input('Data de Início', value=pd.to_datetime('2022-01-01'))
end_date = st.date_input('Data de Fim', value=pd.to_datetime('2024-05-17'))

data = get_data(ticker, start_date, end_date)
data = create_features(data)

st.line_chart(data[['Date', 'Close']].set_index('Date'))

st.write("""
### Previsão para os Próximos 7 Dias
""")

forecast_df = forecast_next_days(model, data)

fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label='Valores Reais')
ax.plot(forecast_df['Date'], forecast_df['Close'], label='Previsões de 7 dias', linestyle='dashed')
ax.legend()
st.pyplot(fig)
