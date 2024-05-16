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
    dftest = adfuller(timeseries['value'].dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    if dfoutput[1] < 0.05:
        print('Conclusão: A série é Estacionária.')
    else:
        print('Conclusão: A série NÃO é Estacionária.')


def import_brent_oil_prices():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from datetime import datetime

    # Função para atualizar o DataFrame com novos dados
    def update_dataframe(df, new_data):
        # Converte a coluna 'Data' para datetime
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
        new_data['Data'] = pd.to_datetime(new_data['Data'], dayfirst=True)

        # Encontra a data mais recente no DataFrame existente
        last_date = df['Data'].max()

        # Filtra as novas linhas que são mais recentes do que a última data
        new_rows = new_data[new_data['Data'] > last_date]

        # Concatena os novos dados com o DataFrame existente se houver novas linhas
        if not new_rows.empty:
            updated_df = pd.concat([df, new_rows], ignore_index=True)
        else:
            updated_df = df

        return updated_df

    # URL do site IPEADATA
    url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

    # Faz uma requisição GET ao site e captura a resposta
    response = requests.get(url)

    # Verifica se a requisição foi bem sucedida
    if response.status_code == 200:
        # Cria um objeto BeautifulSoup para analisar o HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Procura pela tabela no HTML analisado
        table = soup.find('table', {'id': 'grd_DXMainTable'})
        # Usa o pandas para ler a tabela HTML diretamente para um DataFrame
        new_df = pd.read_html(str(table), header=0)[0]

        # Verifica se o arquivo do DataFrame existe e carrega, ou cria um novo DataFrame se não existir
        path = '../Files/ipea.csv'
        try:
            existing_df = pd.read_csv(path)
        except FileNotFoundError:
            existing_df = new_df  # Se o arquivo não existir, considere os dados atuais como o DataFrame existente

        # Atualiza o DataFrame existente com novos dados (carga incremental)
        updated_df = update_dataframe(existing_df, new_df)

        updated_df['Preço - petróleo bruto - Brent (FOB)'] = updated_df['Preço - petróleo bruto - Brent (FOB)']/100

        # Salva o DataFrame atualizado para o arquivo
        updated_df.to_csv(path, index=False)

        # Mostra as primeiras linhas do DataFrame atualizado
        updated_df.head()
    else:
        print('Falha ao acessar a página: Status Code', response.status_code)