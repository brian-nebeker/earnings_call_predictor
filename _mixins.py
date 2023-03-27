import ssl
import certifi
import json
import pickle
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

"""
Mix ins will be used for functions that multiple classes might use
Could be used as a trash heap for storage of in progress functions to keep other code cleaner
"""


class NedHedgeFundMixins:
    def example(self):
        print(2+2)

    def get_api_data(self, url):
        ctx = ssl.create_default_context(cafile=certifi.where())
        response = urlopen(url, context=ctx)
        data = response.read().decode("utf-8")
        return data

    def get_symbols(self):
        # Get symbols
        # https://site.financialmodelingprep.com/developer/docs/stock-market-quote-free-api/#Python
        symbol_url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
        symbol_data = self.get_api_data(symbol_url)
        symbol_df = pd.read_json(symbol_data)
        symbol_df = symbol_df[symbol_df['type'] == 'stock'].copy()
        # symbols = list(symbol_df['symbol'].drop_duplicates())
        return symbol_df

    def get_price_data(self, symbol):
        # Can get multiple symbols but only up to 5, would require a different function
        price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}&from=2000-01-01"
        price_data = self.get_api_data(price_url)
        price_df = pd.read_json(price_data)
        if 'historical' in price_df.columns:
            price_df = price_df['historical'].apply(pd.Series)
            price_df['date'] = pd.to_datetime(price_df['date'])
            return price_df
        else:
            print(f"No historical data found for symbol: {symbol}")
            return pd.DataFrame()

    def calc_target_variables(self, df, column, target_periods):
        for period in target_periods:
            percent_change_column_name = str(column) + '_pct_chg_' + str(period)
            target_variable_column_name = 'target_' + str(column) + '_' + str(period)

            df[percent_change_column_name] = df[column].pct_change(periods=period)
            df[target_variable_column_name] = df[percent_change_column_name].shift(-period)

            df.drop(percent_change_column_name, axis=1, inplace=True)
        return df

    def calc_percent_change(self, df, column, percent_change_periods):
        for period in percent_change_periods:
            df[str(column) + '_pct_chg_' + str(period)] = df[column].pct_change(periods=period)
        return df

    def calc_rolling_avg(self, df, column, rolling_avg_periods):
        for period in rolling_avg_periods:
            df[str(column) + 'rolling' + str(period)] = df[column].rolling(period).mean()
        return df

    def calc_lag_lead(self, df, column, lags):
        for lag in lags:
            df[str(column) + '_lag' + str(lag)] = df[column].shift(lag)
            df[str(column) + '_lead' + str(lag)] = df[column].shift(-lag)
        return df

    def calc_lag(self, df, column, lags):
        for lag in lags:
            df[str(column) + '_lag' + str(lag)] = df[column].shift(lag)
        return df

    def calc_volatility(self, df, column, window):
        df[str(column) + '_std' + str(window)] = df[column].rolling(window).std()
        df[str(column) + '_volatility' + str(window)] = df[column].rolling(window).std() * np.sqrt(window)
        return df

    def calc_rel_strength_index(self, df, column, period):  # above 70 vs below 30
        # Relative Strength Index
        delta = df[column].copy().diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        average_gain = gain.rolling(window=period).mean()
        average_loss = loss.rolling(window=period).mean()

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))

        rsi_upper_check = np.where(rsi > 70, 1, 0)
        rsi_lower_check = np.where(rsi < 30, -1, 0)
        rsi_check = rsi_upper_check + rsi_lower_check

        df[str(column) + '_rsi' + str(period)] = rsi
        df[str(column) + '_rsi_bi' + str(period)] = rsi_check
        return df

    def calc_bollinger_bands(self, df, column, window_size, num_std):  # 2std, 20 window
        # Bollinger Bands
        # Calculate the rolling mean and standard deviation of the closing prices
        rolling_mean = df[column].rolling(window=window_size).mean()
        rolling_std = df[column].rolling(window=window_size).std()

        # Calculate the upper and lower Bollinger Bands
        upper_band = rolling_mean + (num_std * rolling_std)
        lower_band = rolling_mean - (num_std * rolling_std)

        # Add the Bollinger Bands to the dataframe
        df[str(column) + '_boll_upper_w' + str(window_size) + '_std' + str(num_std)] = upper_band
        df[str(column) + '_boll_lower_w' + str(window_size) + '_std' + str(num_std)] = lower_band
        return df

    def calc_macd(self, df, column, short_ema_period, long_ema_period):  # 26/12/9
        # Moving Average Convergence Divergence MACD
        # Calculate the short and long exponential moving averages
        short_ema = df[column].ewm(span=short_ema_period).mean()
        long_ema = df[column].ewm(span=long_ema_period).mean()

        # Calculate the MACD and signal line
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=9).mean()

        # Add the MACD and signal line to the dataframe
        df[str(column) + '_macd' + str(short_ema_period) + '_' + str(long_ema_period)] = macd
        df[str(column) + '_signal_line' + str(short_ema_period) + '_' + str(long_ema_period)] = signal_line
        return df

    def engineer_features(self, df):
        # Percent Changes
        target_periods = [1, 30, 60, 90, 252]
        engineered_df = self.calc_target_variables(df, 'close', target_periods)

        percent_change_periods = [1, 30, 60, 90, 252]
        engineered_df = self.calc_percent_change(engineered_df, 'close', percent_change_periods)

        # Rolling Averages
        rolling_avg_periods = [i for i in range(1, 252, 10)]
        engineered_df = self.calc_rolling_avg(engineered_df, 'close', rolling_avg_periods)
        engineered_df = self.calc_rolling_avg(engineered_df, 'close_pct_chg_1', rolling_avg_periods)

        # Lags
        lags = [i for i in range(1, 252, 10)]
        engineered_df = self.calc_lag(engineered_df, 'close', lags)
        engineered_df = self.calc_lag(engineered_df, 'close_pct_chg_1', lags)

        # Volatility
        vol_windows = [30, 90, 180, 252]
        for window in vol_windows:
            engineered_df = self.calc_volatility(engineered_df, 'close', window)
            engineered_df = self.calc_volatility(engineered_df, 'close_pct_chg_1', window)

        # Relative Strength Index
        rs_periods = [7, 14, 30]
        for period in rs_periods:
            engineered_df = self.calc_rel_strength_index(engineered_df, 'close', period)

        # Bollinger Bands
        engineered_df = self.calc_bollinger_bands(engineered_df, 'close', 10, 1.5)
        engineered_df = self.calc_bollinger_bands(engineered_df, 'close', 20, 2)
        engineered_df = self.calc_bollinger_bands(engineered_df, 'close', 50, 2.5)

        # MACD
        engineered_df = self.calc_macd(engineered_df, 'close', 9, 12)
        engineered_df = self.calc_macd(engineered_df, 'close', 12, 26)
        engineered_df = self.calc_macd(engineered_df, 'close', 9, 26)

        # Set date as index, add suffix to columns
        engineered_df = engineered_df.set_index('date')
        return engineered_df


