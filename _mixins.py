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
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = configuration().api_key

"""
Mix ins will be used for functions that multiple classes might use
Could be used as a trash heap for storage of in progress functions to keep other code cleaner
"""


class NedHedgeFundMixins:
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
            price_df = price_df.sort_index(ascending=False)
            return price_df
        else:
            print(f"No historical data found for symbol: {symbol}")
            return pd.DataFrame()

    def calc_target_variables(self, df, column, target_periods):
        for period in target_periods:
            percent_change_column_name = f"{column}_pct_chg_{period}"
            target_variable_column_name = f"target_{column}_{period}"

            df[percent_change_column_name] = df[column].pct_change(periods=period)
            df[target_variable_column_name] = df[percent_change_column_name].shift(-period)

            df.drop(percent_change_column_name, axis=1, inplace=True)
        return df

    def calc_percent_change(self, df, column, percent_change_periods):
        for period in percent_change_periods:
            df[f"{column}_pct_chg{period}"] = df[column].pct_change(periods=period)
        return df

    def calc_rolling_avg(self, df, column, rolling_avg_periods):
        for period in rolling_avg_periods:
            df[f"{column}_rolling{period}"] = df[column].rolling(period).mean()
        return df

    def calc_lag_lead(self, df, column, lags):
        for lag in lags:
            df[f"{column}_lag{lag}"] = df[column].shift(lag)
            df[f"{column}_lead{lag}"] = df[column].shift(-lag)
        return df

    def calc_lag(self, df, column, lags):
        for lag in lags:
            df[f"{column}_lag{lag}"] = df[column].shift(lag)
        return df

    def calc_volatility(self, df, column, window):
        df[f"{column}_std_w{window}"] = df[column].rolling(window).std()
        df[f"{column}_volatility_w{window}"] = df[column].rolling(window).std() * np.sqrt(window)
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

        df[f"{column}_rsi_p{period}"] = rsi
        df[f"{column}_rsi_check_p{period}"] = rsi_check
        return df

    def calc_bollinger_bands(self, df, column, window_size, num_std):  # 2std, 20 window
        # Bollinger Bands
        # Calculate the rolling mean and standard deviation of the closing prices
        rolling_mean = df[column].rolling(window=window_size).mean()
        rolling_std = df[column].rolling(window=window_size).std()

        # Calculate the upper and lower Bollinger Bands
        upper_band = rolling_mean + (num_std * rolling_std)
        lower_band = rolling_mean - (num_std * rolling_std)

        # Calculate distance of column to bands
        distance_upper_band = upper_band - df[column]
        distance_lower_band = df[column] - lower_band

        # Create check column if column passes band
        upper_check = np.where(distance_upper_band < 0, 1, 0)
        lower_check = np.where(distance_lower_band < 0, -1, 0)
        bollinger_check = upper_check + lower_check

        # Add the Bollinger Bands to the dataframe
        df[f"{column}_boll_upper_w{window_size}_std{num_std}"] = upper_band
        df[f"{column}_boll_lower_w{window_size}_std{num_std}"] = lower_band
        df[f"{column}_boll_dist_upper_w{window_size}_std{num_std}"] = distance_upper_band
        df[f"{column}_boll_dist_lower_w{window_size}_std{num_std}"] = distance_lower_band
        df[f"{column}_boll_check_w{window_size}_std{num_std}"] = bollinger_check
        return df

    def calc_macd(self, df, column, short_ema_period, long_ema_period, signal_period):  # 26/12/9
        # Moving Average Convergence Divergence MACD
        # Calculate the short and long exponential moving averages
        short_ema = df[column].ewm(span=short_ema_period).mean()
        long_ema = df[column].ewm(span=long_ema_period).mean()

        # Calculate the MACD and signal line
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period).mean()

        upper_check = np.where(macd > signal_line, 1, 0)
        lower_check = np.where(macd < signal_line, -1, 0)

        # Add the MACD and signal line to the dataframe
        df[f"{column}_macd_p{short_ema_period}_{long_ema_period}"] = macd
        df[f"{column}_signal_line_p{short_ema_period}_{long_ema_period}"] = signal_line
        return df

    def calc_min_max_scale(self, df, column):
        scaler = MinMaxScaler()
        scaler.fit(df[[column]])
        scaled_column = scaler.transform(df[[column]])
        df[f"{column}_mmscaled"] = scaled_column
        return df

    def calc_engineer_features(self,
                               df,
                               do_target_variable,
                               do_percent_change,
                               do_rolling_avg,
                               do_lags,
                               do_volatility,
                               do_relative_strength,
                               do_bollinger_bad,
                               do_macd,
                               do_min_max_scale,
                               target_periods,
                               percent_change_periods,
                               rolling_avg_periods,
                               lags,
                               volatility_windows,
                               relative_strength_periods,
                               bollinger_band_params,
                               macd_parms,
                               target_variable_col,
                               percent_change_col,
                               rolling_avg_col,
                               lags_col,
                               volatility_col,
                               relative_strength_col,
                               bollinger_band_col,
                               macd_col,
                               min_max_scale_col):

        # Create copy of df as engineered df to avoid UnboundLocalError
        engineered_df = df.copy()


        # Percent Changes
        if do_target_variable:
            for col in target_variable_col:
                engineered_df = self.calc_target_variables(engineered_df, col, target_periods)

        if do_percent_change:
            for col in percent_change_col:
                engineered_df = self.calc_percent_change(engineered_df, col, percent_change_periods)
        # TODO: determine why pct chg volume creates inf and nans

        # Rolling Averages
        if do_rolling_avg:
            for col in rolling_avg_col:
                engineered_df = self.calc_rolling_avg(engineered_df, col, rolling_avg_periods)
        # TODO: determine why rolling volume creates inf and nans

        # Lags
        if do_lags:
            for col in lags_col:
                engineered_df = self.calc_lag(engineered_df, col, lags)
        # TODO: determine why lagged pct chg leads to inf and nans

        # Volatility
        if do_volatility:
            for col in volatility_col:
                for window in volatility_windows:
                    engineered_df = self.calc_volatility(engineered_df, col, window)

        # Relative Strength Index
        if do_relative_strength:
            for col in relative_strength_col:
                for period in relative_strength_periods:
                    engineered_df = self.calc_rel_strength_index(engineered_df, col, period)

        # Bollinger Bands
        if do_bollinger_bad:
            for col in bollinger_band_col:
                for window, std in bollinger_band_params:
                    engineered_df = self.calc_bollinger_bands(engineered_df, col, window, std)

        # MACD
        if do_macd:
            for col in macd_col:
                for lower, upper, signal in macd_parms:
                    engineered_df = self.calc_macd(engineered_df, col, lower, upper, signal)

        # Min Max Scaling
        if do_min_max_scale:
            for col in min_max_scale_col:
                engineered_df = self.calc_min_max_scale(engineered_df, col)

        # Set date as index, add suffix to columns
        # engineered_df = engineered_df.set_index('date')
        return engineered_df

    def engineer_features_old(self, df):
        # Percent Changes
        target_periods = [1, 30, 60, 90, 252]
        engineered_df = self.calc_target_variables(df, 'close', target_periods)

        percent_change_periods = [1, 30, 60, 90, 252]
        engineered_df = self.calc_percent_change(engineered_df, 'close', percent_change_periods)
        engineered_df = self.calc_percent_change(engineered_df, 'volume', percent_change_periods)

        # Rolling Averages
        rolling_avg_periods = [10, 20, 30, 60, 120, 252]
        engineered_df = self.calc_rolling_avg(engineered_df, 'close', rolling_avg_periods)
        engineered_df = self.calc_rolling_avg(engineered_df, 'close_pct_chg1', rolling_avg_periods)
        engineered_df = self.calc_rolling_avg(engineered_df, 'volume', rolling_avg_periods)

        # Lags
        lags = [10, 20, 30, 60, 120, 252]
        engineered_df = self.calc_lag(engineered_df, 'close', lags)
        engineered_df = self.calc_lag(engineered_df, 'close_pct_chg1', lags)

        # Volatility
        vol_windows = [30, 90, 180, 252]
        for window in vol_windows:
            engineered_df = self.calc_volatility(engineered_df, 'close', window)
            engineered_df = self.calc_volatility(engineered_df, 'close_pct_chg1', window)

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
        # engineered_df = engineered_df.set_index('date')
        return engineered_df
