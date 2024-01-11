import finnhub
import pandas as pd
import numpy as np
from datetime import datetime

finnhub_client = finnhub.Client(api_key="Your Key")

#The stock symbol
stock = 'NVDA'
time_interval = '15'
#UNIX timestamp (UTC) fitting 
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 9, 19)
start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())
 
#Transfrom Dict to DataFrame
df1 = finnhub_client.stock_candles(stock, time_interval, start_timestamp, end_timestamp)
df2 = pd.DataFrame(df1)
df2.drop(['s'], axis=1, inplace=True)

#copy&concat
df3 = df2.copy(deep=True)
df4 = df2.copy(deep=True)

for i in range(0, len(df3)-1, 2):
        df3.iloc[i+1] = df3.iloc[i]

for i in range(0, len(df4)-1, 4):
        for j in range (1, 4, 1):
            df4.iloc[i+j] = df4.iloc[i]

df = pd.concat([df3,df2,df4], axis=1)

#Save as csv
print(df)
df.to_csv(f'data/{stock}_candles_{time_interval}_concat.csv', index=False)