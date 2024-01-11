import finnhub
finnhub_client = finnhub.Client(api_key="Your Key")

print(finnhub_client.symbol_lookup('sp500'))
