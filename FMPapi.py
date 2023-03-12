import ssl
import certifi
import json
import pandas as pd
from urllib.request import urlopen
from tqdm import tqdm
from config_file import configuration

api_key = configuration().api_key


def get_api_data(url):
    ctx = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=ctx)
    data = response.read().decode("utf-8")
    return data


def get_symbols():
    # Get symbols
    # https://site.financialmodelingprep.com/developer/docs/stock-market-quote-free-api/#Python
    symbol_url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
    symbol_data = get_api_data(symbol_url)
    symbol_df = pd.read_json(symbol_data)
    symbol_df = symbol_df[symbol_df['type'] == 'stock'].copy()
    #symbols = list(symbol_df['symbol'].drop_duplicates())
    return symbol_df


def get_price_data(symbol):
    # Get price data
    # https://site.financialmodelingprep.com/developer/docs/historical-stock-data-free-api/#Python
    price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}&from=1000-01-01"
    price_data = get_api_data(price_url)
    price_df = pd.read_json(price_data)
    price_df = price_df['historical'].apply(pd.Series)
    price_df['date'] = pd.to_datetime(price_df['date'])

    # Add weekends into data set
    date_range = pd.DataFrame(pd.date_range(start=price_df['date'].min(), end=price_df['date'].max()), columns=['date'])
    date_range.sort_values('date', ascending=False, inplace=True)
    price_df = pd.merge(date_range, price_df, on='date', how='left')

    # Calculate quarters and years for price history
    price_df['quarter'] = price_df['date'].dt.quarter
    price_df['year'] = price_df['date'].dt.year

    # Drop unecessary columns
    price_drop_columns = ['change', 'changePercent', 'vwap', 'label', 'changeOverTime']
    price_df.drop(price_drop_columns, axis=1, inplace=True)

    # Backward fill nans in price dataframe
    price_df = price_df.fillna(method='bfill')

    # Lag and lead variables
    columns_to_shift = ['open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume']
    lags = [360, 180, 90, 30, 1, -1, -30, -90, -180, -360]
    for lag in lags:
        for column in columns_to_shift:
            # TODO: Remove negatives from names, if statement to state lag or lead
            price_df[str(lag) + '_lag_' + str(column)] = price_df[column].shift(lag)
    return price_df


def combine_earnings_price(symbol, price_df):
    # Collect all earnings transcripts
    # https://site.financialmodelingprep.com/developer/docs/earning-call-transcript-api/#Python
    all_years_quarters = price_df[['year', 'quarter']].copy().drop_duplicates()
    all_earnings_df = pd.DataFrame()
    previous_blank = False
    for index, row in all_years_quarters.iterrows():
        # Get quarter and year from the earliest date in price data
        quarter = row['quarter']
        year = row['year']

        # Create url
        earnings_url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?quarter={quarter}&year={year}&apikey={api_key}"

        # Create and adjust dataframe for earnings data
        earn_data = get_api_data(earnings_url)
        if earn_data == '[]':  # Check for empty api call
            if previous_blank:  # Check if 2 empty api calls happened back to back
                if all_earnings_df.empty:  # If no api call succeeded return nothing
                    return
                else:
                    try:  # Try to merge earnings and price
                        result = pd.merge(all_earnings_df, price_df, on='date', how='inner')
                        return result
                    except KeyError:  # Sometimes data miss match, if so return an empty df
                        result = pd.DataFrame()
                        return result
            previous_blank = True
            continue
        earn_df = pd.read_json(earn_data)
        earn_df['time'] = pd.to_datetime(earn_df['date']).dt.time
        earn_df['date'] = pd.to_datetime(earn_df['date'].dt.date)
        # earn_df['date'] = pd.to_datetime(earn_df['date'])
        all_earnings_df = pd.concat([all_earnings_df, earn_df], ignore_index=True)
        previous_blank = False

    result = pd.merge(all_earnings_df, price_df, on='date', how='inner')
    return result


symbol_df = get_symbols()
symbol_df = symbol_df.loc[(symbol_df['exchangeShortName'] == 'NYSE') | (symbol_df['exchangeShortName'] == 'NASDAQ')]
symbol_df = symbol_df[symbol_df['price']>0]
symbols = list(symbol_df['symbol'].drop_duplicates())
data = []  # Switched to list of dataframes for faster run time
for symbol in tqdm(symbols, desc="Collecting data"):
    price_df = get_price_data(symbol)
    result = combine_earnings_price(symbol, price_df)
    # TODO: Clean up df to remove same names before merge
    data.append(result)
    #data = pd.concat([data, result], ignore_index=True)

# concats all dfs
df = pd.concat(data, ignore_index=True)
data.to_parquet('initial_data.prq')


# TODO: engineer features to act as inputs or targets, % change as it relates to broader market
# TODO: look at how a stock price changes vs VTI, if it increases 10% but the rest of the market avg'ed at +20% its not as good
# TODO: categorize stocks, could be useful for forced diversification
