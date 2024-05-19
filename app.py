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
from funcs import get_data, load_data, load_gdp_data, load_sp500_data, create_features, forecast_next_days

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
his_data = load_data()
his_data = create_features(his_data)
gdp_data = load_gdp_data()
sp500_data = load_sp500_data()
ticker = 'BZ=F'
start_date = '1987-01-01'
data_y = get_data(ticker, start_date, end_date)
data_y = create_features(data_y) 
forecast_df = forecast_next_days(model, data_y)
last_7_days = forecast_df.tail(7)


st.title("Análise e Previsão de Preços do Petróleo Brent")
st.subheader("Dashboard Interativo com Insights Econômicos e Geopolíticos")

tabs = st.tabs(["Dashboard", "Forecasting"])



with tabs[0]:
    st.header("Dashboard de Insights")
    st.divider()
    st.write("""
    ### Impacto de Crises Econômicas
    A crise financeira de 2008, impactou o preço do petróleo Brent. Marcada pela falência do Lehman Brothers, resultou em uma queda significativa nos preços do petróleo.
    """)

    fig_crisis = go.Figure()
    fig_crisis.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço Histórico'))
    fig_crisis.add_vrect(x0="2008-09-01", x1="2009-03-01", fillcolor="red", opacity=0.5, line_width=0, annotation_text="Crise Financeira 2008", annotation_position="top left")
    fig_crisis.update_layout(xaxis_title='Data',
                            yaxis_title='Preço (USD)',
                            xaxis=dict(rangeslider=dict(visible=True)))

    container = st.container(border=True)
    container.plotly_chart(fig_crisis, use_container_width=True)

    st.write("""
    ### Efeitos de Situações Geopolíticas
    Situações geopolíticas, como conflitos no Oriente Médio, têm um impacto significativo nos preços do petróleo. Eventos como a Guerra do Golfo e a Primavera Árabe.
    """)

    fig_geopolitical = go.Figure()
    fig_geopolitical.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço Histórico'))
    fig_geopolitical.add_vrect(x0="1990-08-01", x1="1991-02-28", fillcolor="orange", opacity=0.5, line_width=0, annotation_text="Guerra do Golfo", annotation_position="top left")
    fig_geopolitical.add_vrect(x0="2010-12-17", x1="2011-12-17", fillcolor="purple", opacity=0.5, line_width=0, annotation_text="Primavera Árabe", annotation_position="top left")
    fig_geopolitical.update_layout(xaxis_title='Data',
                                yaxis_title='Preço (USD)',
                                xaxis=dict(rangeslider=dict(visible=True)))

    container = st.container(border=True)
    container.plotly_chart(fig_geopolitical, use_container_width=True)

    st.write("""
    ### Demanda Global por Energia
    A demanda global por energia, especialmente em países em desenvolvimento, afeta diretamente os preços do petróleo. Aqui comparamos o crescimento econômico com a variação dos preços do petróleo.
    """)

    his_data['Year'] = his_data['Date'].dt.year
    merged_data = pd.merge(his_data, gdp_data, left_on='Year', right_on='Year', how='left')

    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Close'], mode='lines', name='Preço do Petróleo'))
    fig_demand.add_trace(go.Scatter(x=merged_data['Date'], y=merged_data['Global GDP'], mode='lines', name='PIB Global (%)', yaxis='y2'))

    fig_demand.update_layout(
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        yaxis2=dict(title='PIB Global (%)', overlaying='y', side='right'),
        xaxis=dict(rangeslider=dict(visible=True))
    )

    container = st.container(border=True)
    container.plotly_chart(fig_demand, use_container_width=True)

    st.write("""
    ### Comparação com Outros Indicadores Econômicos
    Comparativo da variação do preço do petróleo com o índice S&P 500, para entender melhor a correlação.
    """)

    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(x=his_data['Date'], y=his_data['Close'], mode='lines', name='Preço do Petróleo'))
    fig_comparison.add_trace(go.Scatter(x=sp500_data['Date'], y=sp500_data['Close'], mode='lines', name='S&P 500'))

    fig_comparison.update_layout(
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        yaxis_type="log",
        xaxis=dict(rangeslider=dict(visible=True))
    )

    container = st.container(border=True)
    container.plotly_chart(fig_comparison, use_container_width=True)
    st.divider()

with tabs[1]:
    st.header("Previsão de Preços do Petróleo Brent")    
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_7_days['Date'], y=last_7_days['Close'], mode='lines+markers+text', name='Previsões de 7 dias', line=dict(dash='dash'),
                            text=[f"{value:.2f}" for value in last_7_days['Close']],
                            textposition="top center"))

    fig.update_layout(xaxis_title='Data',
                    yaxis_title='US$')

    container = st.container(border=True)
    container.plotly_chart(fig, use_container_width=True)

    st.divider()




