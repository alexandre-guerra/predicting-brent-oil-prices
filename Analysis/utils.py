def test_stationarity(timeseries):
    import pandas as pd
    import plotly.graph_objs as go
    from statsmodels.tsa.stattools import adfuller

    timeseries.index = pd.to_datetime(timeseries.index)
    movingAverage = timeseries['value'].rolling(window=12).mean()
    movingSTD = timeseries['value'].rolling(window=12).std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries['value'], mode='lines', name='Original',  line=dict(color='rgba(30, 144, 255, 0.8)', width=2)))

    fig.add_trace(go.Scatter(x=movingAverage.index, y=movingAverage, mode='lines', name='Rolling Mean', line=dict(color='rgba(255, 99, 71, 0.8)', width=2)))

    fig.add_trace(go.Scatter(x=movingSTD.index, y=movingSTD, mode='lines', name='Rolling Std', line=dict(color='rgba(44, 44, 44, 0.8)', width=2)))

    fig.update_layout(
        title='Rolling Mean & Standard Deviation',
        xaxis_title='Date',
        yaxis_title='Values',
        height=600,
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            type='date'
        ),
        legend_title='Legend'
    )

    fig.show()

    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    if dfoutput[1] < 0.05:
        print('Conclusão: A série é Estacionária.')
    else:
        print('Conclusão: A série NÃO é Estacionária.')
