import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join('.', 'Files')))
from utils import get_data, load_data, load_gdp_data, load_sp500_data, create_features, forecast_next_days

st.set_page_config(
    page_title="Brent Oil Forecasting App",
    layout="wide",
    page_icon="💸"
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

st.markdown(f"""
<style>
div[data-testid="column"] {{
  text-align: center;
}}
</style>
""", unsafe_allow_html=True)

model = joblib.load('./Files/xgboost_model.pkl')
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = '1987-01-01'

st.title('Previsão de Preços do Petróleo Brent')
st.divider()

st.write("""
### Resultados do Modelo XGBoost
""")

col1, col2, col3 = st.columns(3)

with col1:
    container = st.container(border=True)
    container.metric(label="Mean Absolute Percentage Error (MAPE)", value="0.02 %", delta=None)
with col2:
    container = st.container(border=True)
    container.metric(label="Root Mean Squared Error (RMSE)", value="1.85", delta=None)
with col3:
    container = st.container(border=True)
    container.metric(label="Mean Absolute Error (MAE)", value="1.27", delta=None)



st.write("""
### Dados Históricos
""")

col1, col2 = st.columns(2)
with col1:
    container = st.container(border=True)
    container.write(f"**Data Inicial:** {start_date}")
with col2:
    container = st.container(border=True)
    container.write(f"**Data Final:** {end_date}")

his_data = load_data()
his_data = create_features(his_data)

fig_historical = go.Figure()
fig_historical.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Valores Reais'))

fig_historical.update_layout(
    xaxis_title='Data',
    yaxis_title='US$',
    xaxis=dict(rangeslider=dict(visible=True), type="date")
)

container = st.container(border=True)
container.plotly_chart(fig_historical, use_container_width=True)


st.write("""
### Previsão para os Próximos 7 Dias
""")

ticker = 'BZ=F'
data_y = get_data(ticker, start_date, end_date)
data_y = create_features(data_y) 
forecast_df = forecast_next_days(model, data_y)
last_7_days = forecast_df.tail(7)

fig = go.Figure()
fig.add_trace(go.Scatter(x=last_7_days['Date'], y=last_7_days['Close'], mode='lines+markers+text', name='Previsões de 7 dias', line=dict(dash='dash'),
                         text=[f"{value:.2f}" for value in last_7_days['Close']],
                         textposition="top center"))

fig.update_layout(xaxis_title='Data',
                  yaxis_title='US$')

container = st.container(border=True)
container.plotly_chart(fig, use_container_width=True)

st.divider()
gdp_data = load_gdp_data()

# Insight 1: Impacto de Crises Econômicas
st.write("""
### Insight 1: Impacto de Crises Econômicas
Exploramos como grandes crises econômicas, como a crise financeira de 2008, impactaram o preço do petróleo Brent. A crise de 2008, marcada pela falência do Lehman Brothers, resultou em uma queda significativa nos preços do petróleo.
""")

fig_crisis = go.Figure()
fig_crisis.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço Histórico'))
fig_crisis.add_vrect(x0="2008-09-01", x1="2009-03-01", fillcolor="red", opacity=0.5, line_width=0, annotation_text="Crise Financeira 2008", annotation_position="top left")
fig_crisis.update_layout(title='Impacto da Crise Financeira de 2008',
                         xaxis_title='Data',
                         yaxis_title='Preço (USD)',
                         xaxis=dict(rangeslider=dict(visible=True)))

st.plotly_chart(fig_crisis, use_container_width=True)

# Insight 2: Efeitos de Situações Geopolíticas
st.write("""
### Insight 2: Efeitos de Situações Geopolíticas
Situações geopolíticas, como conflitos no Oriente Médio, têm um impacto significativo nos preços do petróleo. Examinamos eventos como a Guerra do Golfo e a Primavera Árabe.
""")

fig_geopolitical = go.Figure()
fig_geopolitical.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço Histórico'))
fig_geopolitical.add_vrect(x0="1990-08-01", x1="1991-02-28", fillcolor="orange", opacity=0.5, line_width=0, annotation_text="Guerra do Golfo", annotation_position="top left")
fig_geopolitical.add_vrect(x0="2010-12-17", x1="2011-12-17", fillcolor="purple", opacity=0.5, line_width=0, annotation_text="Primavera Árabe", annotation_position="top left")
fig_geopolitical.update_layout(title='Efeitos de Situações Geopolíticas',
                               xaxis_title='Data',
                               yaxis_title='Preço (USD)',
                               xaxis=dict(rangeslider=dict(visible=True)))

st.plotly_chart(fig_geopolitical, use_container_width=True)

# Insight 3: Demanda Global por Energia
st.write("""
### Insight 3: Demanda Global por Energia
A demanda global por energia, especialmente em países em desenvolvimento, afeta diretamente os preços do petróleo. Aqui comparamos o crescimento econômico com a variação dos preços do petróleo.
""")

# Mesclar os dados do PIB global com os dados do petróleo
his_data['Year'] = his_data['Date'].dt.year
merged_data = pd.merge(his_data, gdp_data, left_on='Year', right_on='Year', how='left')

fig_demand = go.Figure()
fig_demand.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Close'], mode='lines', name='Preço do Petróleo'))
fig_demand.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Global GDP'], mode='lines', name='PIB Global (%)', yaxis='y2'))

fig_demand.update_layout(
    title='Demanda Global por Energia e Preços do Petróleo',
    xaxis_title='Data',
    yaxis_title='Preço (USD)',
    yaxis2=dict(title='PIB Global (%)', overlaying='y', side='right'),
    xaxis=dict(rangeslider=dict(visible=True))
)

st.plotly_chart(fig_demand, use_container_width=True)

# Carregar os dados do S&P 500
sp500_data = load_sp500_data()

# Insight 4: Comparação com Outros Indicadores Econômicos
st.write("""
### Insight 4: Comparação com Outros Indicadores Econômicos
Comparamos a variação do preço do petróleo com outros indicadores econômicos, como o índice S&P 500 e as taxas de câmbio, para entender melhor as correlações.
""")

# Gráfico comparando preços do petróleo com S&P 500
fig_comparison = go.Figure()
fig_comparison.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço do Petróleo'))
fig_comparison.add_trace(go.Scatter(x=sp500_data['Date'], y=sp500_data['Close'], mode='lines', name='S&P 500'))

fig_comparison.update_layout(
    title='Comparação com Outros Indicadores Econômicos',
    xaxis_title='Data',
    yaxis_title='Preço (USD)',
    xaxis=dict(rangeslider=dict(visible=True))
)

st.plotly_chart(fig_comparison, use_container_width=True)
