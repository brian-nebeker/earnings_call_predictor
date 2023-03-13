import ssl
import certifi
import json
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import date
from urllib.request import urlopen
from tqdm import tqdm
from config_file import configuration
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # symbols = list(symbol_df['symbol'].drop_duplicates())
    return symbol_df


def get_price_data(symbol):
    # Can get multiple symbols but only up to 5, would require a different function
    price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}&from=2000-01-01"
    price_data = get_api_data(price_url)
    price_df = pd.read_json(price_data)
    if 'historical' in price_df.columns:
        price_df = price_df['historical'].apply(pd.Series)
        price_df['date'] = pd.to_datetime(price_df['date'])
        return price_df
    else:
        print(f"No historical data found for symbol: {symbol}")
        return pd.DataFrame()


def calc_percent_change(df, percent_change_periods):
    for period in percent_change_periods:
        df['percent_change_' + str(period)] = df['close'].pct_change(periods=period)
        df['target_percent_change_' + str(period)] = df['percent_change_' + str(period)].shift(-period)
    return df


def calc_rolling_avg(df, rolling_avg_periods):
    for period in rolling_avg_periods:
        df['rolling' + str(period)] = df['close'].rolling(period).mean()
    return df


def calc_lag_lead(df, lags):
    for lag in lags:
        df['close_lag_' + str(lag)] = df['close'].shift(lag)
        df['close_lead_' + str(lag)] = df['close'].shift(-lag)
    return df


def calc_volatility(df, window):
    df['std252'] = df['close'].rolling(window).std()
    df['annual_volatility'] = df['close'].rolling(window).std() * np.sqrt(window)
    return df


def calc_rel_strength_index(df, period):
    # Relative Strength Index
    delta = df['close'].copy().diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    return df


def calc_bollinger_bands(df, window_size, num_std):
    # Bollinger Bands
    # Calculate the rolling mean and standard deviation of the closing prices
    rolling_mean = df['close'].rolling(window=window_size).mean()
    rolling_std = df['close'].rolling(window=window_size).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    # Add the Bollinger Bands to the dataframe
    df['bollinger_upper'] = upper_band
    df['bollinger_lower'] = lower_band
    return df


def calc_macd(df, short_ema_period, long_ema_period):
    # Moving Average Convergence Divergence MACD
    # Calculate the short and long exponential moving averages
    short_ema = df['close'].ewm(span=short_ema_period).mean()
    long_ema = df['close'].ewm(span=long_ema_period).mean()

    # Calculate the MACD and signal line
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9).mean()

    # Add the MACD and signal line to the dataframe
    df['macd'] = macd
    df['signal_line'] = signal_line
    return df


def engineer_features(symbol):
    # Optional delay to ensure 300 api calls per minute is not exceeded
    # time.sleep(0.2)

    # Get price data
    price_df = get_price_data(symbol)
    if price_df.empty:
        return price_df

    # Get Price Data Ready
    engineered_df = price_df[['date', 'close']].copy()
    engineered_df.sort_values(by='date', inplace=True)

    # Feature Engineering
    # Percent Changes
    percent_change_periods = [1, 30, 60, 90, 252]
    engineered_df = calc_percent_change(engineered_df, percent_change_periods)

    # Rolling Averages
    rolling_avg_periods = [20, 50, 200]
    engineered_df = calc_rolling_avg(engineered_df, rolling_avg_periods)

    # Lags and Leads
    lags = [1, 30, 90, 180, 252]
    engineered_df = calc_lag_lead(engineered_df, lags)

    # Volatility
    vol_window = 252
    engineered_df = calc_volatility(engineered_df, vol_window)

    # Relative Strength Index
    rs_period = 14
    engineered_df = calc_rel_strength_index(engineered_df, rs_period)

    # Bollinger Bands
    bollinger_window = 20
    bollinger_std = 2
    engineered_df = calc_bollinger_bands(engineered_df, bollinger_window, bollinger_std)

    # MACD
    short_ema_period = 12
    long_ema_period = 26
    engineered_df = calc_macd(engineered_df, short_ema_period, long_ema_period)

    # Set date as index, add suffix to columns
    engineered_df = engineered_df.set_index('date')
    #engineered_df = engineered_df.add_suffix('_'+str(symbol))
    return engineered_df


def engineer_features_wrapper(symbol):
    # Wrap engineer features, return symbol and dataframe as tuple
    df = engineer_features(symbol)
    return (symbol, df)


# Initiate start time
start_time = time.time()
print("Start")

# Create symbols for dataframe construction
symbols = get_symbols()
symbols = symbols[symbols['exchangeShortName'] == 'NYSE'].copy()
symbols = symbols[symbols['price']>10].copy()
symbols_list = symbols['symbol'].tolist()
symbols_list = symbols_list[:100]

# Report time to create symbols
end_time = time.time()
print(f"Created symbols. Elapsed:", end_time - start_time)


# Submit jobs for each symbol, future is result of job
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(engineer_features_wrapper, symbol) for symbol in symbols_list]

# Report time to submit jobs
end_time = time.time()
print(f"Submitted jobs. Elapsed:", end_time - start_time)


# Use as_completed to iterate over completed futures store results in dictionary
results_dict = {}
for future in as_completed(futures):
    symbol, df = future.result()
    results_dict[symbol] = df

# Report time to complete jobs
end_time = time.time()
print(f"Completed jobs. Elapsed:", end_time - start_time)


# Combine dataframes into single dataframe
combined_df = pd.concat(results_dict.values(), axis=1, keys=results_dict.keys())


# Report time to complete data creation
end_time = time.time()
print(f"Completed. Elapsed time for {len(symbols_list)} symbols: {(end_time - start_time) / 60}min")


# Best signals
vert_combined_df = pd.concat(results_dict.values(), axis=0)
vert_combined_df = vert_combined_df.dropna()

# Grab target features
targets = vert_combined_df[['target_percent_change_1', 'target_percent_change_30', 'target_percent_change_60',
                           'target_percent_change_90', 'target_percent_change_252']].copy()

for column in targets.columns:
    targets[column+'_T10'] = (targets[column].rank(pct=True)>0.9).astype(int)


# Create independent variables
drop_columns = ['target_percent_change_1', 'target_percent_change_30', 'target_percent_change_60',
                'target_percent_change_90', 'target_percent_change_252', 'close_lead_1',
                'close_lead_30', 'close_lead_90', 'close_lead_180', 'close_lead_252']
X = vert_combined_df.drop(drop_columns, axis=1)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, targets['target_percent_change_1'], test_size=0.25)

# Create model
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=20, verbose=1, n_jobs=-1)

print("Fitting Model")
regr.fit(X_train, y_train)

regr.score(X_test, y_test)

feature_importance = pd.DataFrame(X.columns)
feature_importance['importances'] = regr.feature_importances_

pd.qcut(target3, q=10)



vti = "VTI"
# Index feature engineering
index_df = index_df[['date', 'close']].copy()
index_df.sort_values(by='date', inplace=True)

# Percent Changes
for period in percent_change_periods:
    index_df['percent_change_'+str(period)] = index_df['close'].pct_change(periods=period)

# Add suffix to
index_df = index_df.rename(columns={c: c+'_index' for c in index_df.columns if c not in ['date']})

# Combine df with index dataframe
df = pd.merge(df, index_df, on='date', how='inner')
df.dropna(inplace=True)

# Calculate percent differences between the stock and the index
for period in percent_change_periods:
    df['diff_pc_'+str(period)] = df['percent_change_' + str(period)] - df['percent_change_' + str(period) + '_index']

temp = index_df.describe()


