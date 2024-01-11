import yfinance as yf

start_date = "2000-01-01"
end_date = "2023-11-01"
stock = "^GSPC"

data = yf.download(stock, start=start_date, end=end_date)

data = data.reset_index()

print(data)


data.to_csv(f'data/{stock}_candles_D.csv', index=False)
