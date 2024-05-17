import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go


st.set_page_config(
    page_title="Brent Oil Forecasting App",
    layout="wide",
    page_icon="☕"
)

st.markdown(""" <style>
footer {visibility: hidden;}
h1 {text-align: center;}
</style> """, unsafe_allow_html=True)

st.markdown(f""" <style>
    .appview-container .main .block-container{{
        padding-top: {0}rem;
        padding-right: {1.5}rem;
        padding-left: {1.5}rem;
        padding-bottom: {0}rem;
    }} </style> """, unsafe_allow_html=True)

st.markdown(f"""
<style>
div[data-testid="stMetric"] {{
  padding-top: {0}rem;
        padding-right: {5}rem;
        padding-left: {5}rem;
        padding-bottom: {0}rem;
}}
</style>
""", unsafe_allow_html=True)

# Carregar o modelo treinado
model = joblib.load('xgboost_model.pkl')
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = '1987-01-01'

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
st.divider()

st.write("""
### Dados Históricos
""")

ticker = 'BZ=F'

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Data de Início:** {start_date}")
with col2:
    st.write(f"**Data de Fim:** {end_date}")

data = get_data(ticker, start_date, end_date)
data = create_features(data)

fig_historical = go.Figure()
fig_historical.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Valores Reais'))

fig_historical.update_layout(title='Dados Históricos do Petróleo Brent',
                             xaxis_title='Data',
                             yaxis_title='Preço')

st.plotly_chart(fig_historical, use_container_width=True)

st.divider()

st.write("""
### Previsão para os Próximos 7 Dias
""")

forecast_df = forecast_next_days(model, data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Valores Reais'))
fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name='Previsões de 7 dias', line=dict(dash='dash')))

fig.update_layout(title='Previsões de 7 dias',
                  xaxis_title='Data',
                  yaxis_title='Preço')

st.plotly_chart(fig)
