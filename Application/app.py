import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'brent_oil_model.h5')
# Carregar o modelo usando Keras
predict_model = load_model(model_path)


sequence_length = 7
scaler_pred = MinMaxScaler()
scaler = MinMaxScaler()
csv_path = os.path.join(current_directory, '../Files/oil_prices.csv')
df = pd.read_csv(csv_path,sep=';')
df['period'] = pd.to_datetime(df['period'])
df.head()

df_temp = df[:sequence_length]
df_temp.set_index('period', inplace=True)
new_df = df_temp.sort_values(by=['period']).copy()

last_N_days = new_df.values

last_N_days_scaled = scaler.transform(last_N_days)

X_test_new = []
X_test_new.append(last_N_days_scaled)

pred_price_scaled = predict_model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

price_today = np.round(new_df['value'][-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

plus = '+'; minus = ''
print(f'O valor do Petróleo Brent para último dia disponível {price_today}')
print(f'O valor do Petróleo Brent predito é {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')