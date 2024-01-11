import finnhub
import pandas as pd
import numpy as np
from datetime import datetime

finnhub_client = finnhub.Client(api_key="Your Key")

#The stock symbol
stock = 'WSPX.MI'
time_interval = 'D'

#UNIX timestamp (UTC) fitting 
start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 8, 1)
start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())
 
#Transfrom Dict to DataFrame
df1 = finnhub_client.stock_candles(stock, time_interval , start_timestamp, end_timestamp)
df = pd.DataFrame(df1)
time = df['t'].apply(lambda x: datetime.utcfromtimestamp(x))
date = time.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
df.insert(0,'date',date)
df.drop(['s','t'], axis=1, inplace=True)
#Save as csv
print(df)
df.to_csv(f'data/{stock}_candles_{time_interval}.csv', index=False)