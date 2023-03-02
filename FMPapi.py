import ssl
import certifi
import json
import pandas as pd
from urllib.request import urlopen

api_key = "3dfdfba6272ec8469270c1f8977e726f"
symbol = "AAPL"
quarter = 3
year = 2020

symbol_url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
earnings_url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?quarter={quarter}&year={year}&apikey={api_key}"
price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}&from=1000-01-01"


def get_data(url):
    ctx = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=ctx)
    data = response.read().decode("utf-8")
    return data


# Get symbols
symbol_data = get_data(symbol_url)
symbol_df = pd.read_json(symbol_data)
symbol_df = symbol_df[symbol_df['type'] == 'stock']
len(symbol_df)


# Get price data
price_data = get_data(price_url)
price_df = pd.read_json(price_data)
price_df = price_df['historical'].apply(pd.Series)
price_df['date'] = pd.to_datetime(price_df['date'])

# Add weekends into data set
date_range = pd.DataFrame(pd.date_range(start=price_df['date'].min(), end=price_df['date'].max()), columns=['date'])
date_range.sort_values('date', ascending=False, inplace=True)
price_df = pd.merge(date_range, price_df, on='date', how='left')

# Drop unecessary columns
price_drop_columns = ['change', 'changePercent', 'vwap', 'label', 'changeOverTime']
price_df.drop(price_drop_columns, axis=1, inplace=True)

# Backward fill nans in price dataframe
price_df = price_df.fillna(method='bfill')

# Lag and lead variables
columns_to_shift = ['open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume']
lags = [1, -1, -30, -90, -180, -360]
for lag in lags:
    for column in columns_to_shift:
        price_df[str(lag) + '_lag_' + str(column)] = price_df[column].shift(lag)


# Get earnings transcript data
earn_data = get_data(earnings_url)
earn_df = pd.read_json(earn_data)
earn_df['time'] = pd.to_datetime(earn_df['date']).dt.time
earn_df['date'] = pd.to_datetime(earn_df['date']).dt.date
earn_df['date'] = pd.to_datetime(earn_df['date'])

# Find date of earnings call
earnings_call_date = earn_df.iloc[0]['date']
price_df[price_df['date'] == earnings_call_date]

# Create date range to make up for weekends with missing data
date_range = pd.DataFrame(pd.date_range(start=price_df['date'].min(), end=price_df['date'].max()), columns=['date'])

# Combine date_range df with price_df and earn_df
result = pd.merge(date_range, price_df, on='date', how='left')
result2 = pd.merge(result, earn_df, on='date', how='left')
