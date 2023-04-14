import time
import pickle
import datetime
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from _mixins import NedHedgeFundMixins
tqdm.pandas()


class NedHedgeFund(NedHedgeFundMixins):
    def __init__(self,
                 exchange_name="NYSE",
                 query_price_data: bool = False,
                 do_engineer_features: bool = False,
                 do_univariate_test: bool = False,
                 do_tune_models: bool = False,
                 do_target_variable: bool = True,
                 do_percent_change: bool = True,
                 do_rolling_avg: bool = True,
                 do_lags: bool = True,
                 do_volatility: bool = True,
                 do_relative_strength: bool = True,
                 do_bollinger_bad: bool = True,
                 do_macd: bool = True,
                 do_min_max_scale: bool = True,
                 do_add_index: bool = True,
                 do_engineer_index: bool = True,
                 target_periods: list = [90],
                 percent_change_periods: list = [1, 30, 60, 90, 252],
                 rolling_avg_periods: list = [10, 20, 30, 60, 120, 252],
                 lags: list = [10, 20, 30, 60, 120, 252],
                 volatility_windows: list = [30, 90, 180, 252],
                 relative_strength_periods: list = [7, 14, 30],
                 bollinger_band_params: list = [(10, 1.5), (20, 2), (50, 2.5)],
                 macd_params: list = [(12, 26, 9)],
                 target_variable_col: list = ['close'],
                 percent_change_col: list = ['close'],
                 rolling_avg_col: list = ['close', 'close_pct_chg1'],
                 lags_col: list = ['close', 'close_pct_chg1'],
                 volatility_col: list = ['close', 'close_pct_chg1'],
                 relative_strength_col: list = ['close'],
                 bollinger_band_col: list = ['close'],
                 macd_col: list = ['close'],
                 min_max_scale_col: list = ['close', 'volume']
                 ):
        self.exchange_name = exchange_name
        self.query_price_data = query_price_data
        self.do_engineer_features = do_engineer_features
        self.do_univariate_test = do_univariate_test
        self.do_tune_models = do_tune_models
        self.target_periods = target_periods
        self.percent_change_periods = percent_change_periods
        self.rolling_avg_periods = rolling_avg_periods
        self.lags = lags
        self.volatility_windows = volatility_windows
        self.relative_strength_periods = relative_strength_periods
        self.bollinger_band_params = bollinger_band_params
        self.macd_params = macd_params
        self.do_target_variable = do_target_variable
        self.do_percent_change = do_percent_change
        self.do_rolling_avg = do_rolling_avg
        self.do_lags = do_lags
        self.do_volatility = do_volatility
        self.do_relative_strength = do_relative_strength
        self.do_bollinger_bad = do_bollinger_bad
        self.do_macd = do_macd
        self.do_min_max_scale = do_min_max_scale
        self.do_add_index = do_add_index
        self.do_engineer_index = do_engineer_index
        self.target_variable_col = target_variable_col
        self.percent_change_col = percent_change_col
        self.rolling_avg_col = rolling_avg_col
        self.lags_col = lags_col
        self.volatility_col = volatility_col
        self.relative_strength_col = relative_strength_col
        self.bollinger_band_col = bollinger_band_col
        self.macd_col = macd_col
        self.min_max_scale_col = min_max_scale_col
        self.symbols_list = None
        self.repeat_data_keys = None
        self.sparse_data_keys = None
        self.low_volume_keys = None
        self.use_df = None
        self.price_df = None
        self.call_df = None
        self.price_dict = None
        self.call_dict = None
        self.model = None
        self.tuned_params = None
        self.call_dict_location = f'./assets/data/{self.exchange_name}_earnings_dictionary.pickle'
        self.price_dict_location = f'./assets/data/{self.exchange_name}_price_dictionary.pickle'

    def execute(self):
        # If true, generate all new price data and convert to vert stacked dataframe, if false load dict and convert
        # Query all new price data
        if self.query_price_data:
            print("EXECUTE: Querying new data")
            self.get_symbols_list()

            # Get dictionaries
            self.price_dict = self.get_data_dict('price')
            self.call_dict = self.get_data_dict('call')

            # Match dictionaries
            self.price_dict, self.call_dict = self.match_price_call_keys(self.price_dict, self.call_dict)

            # Transform to dataframes
            self.price_df = self.data_dict_to_dataframe(self.price_dict)
            self.call_df = self.data_dict_to_dataframe(self.call_dict)
        else:
            print("EXECUTE: Loading previous data")
            # Load dictionaries
            self.price_dict = self.load_data_dict('price')
            self.call_dict = self.load_data_dict('call')

            # Match dictionaries
            self.price_dict, self.call_dict = self.match_price_call_keys(self.price_dict, self.call_dict)

            # Convert to dataframes
            self.price_df = self.data_dict_to_dataframe(self.price_dict)
            self.call_df = self.data_dict_to_dataframe(self.call_dict)

        # Engineer features for stacked dataframe
        if self.do_engineer_features:
            print("EXECUTE: Engineering features")
            self.use_df = self.engineer_price_data_features(self.price_df)
        else:
            self.use_df = self.price_df.copy()

        # Add index and engineer index check
        if self.do_add_index:
            print("EXECUTE: Add index")
            self.use_df = self.add_index_features(self.use_df)

        # Combine call and use df
        self.use_df = self.combine_call_price_data(self.call_df, self.use_df)

        # Dump use_df
        print("EXECUTE: dumping dataframe")
        self.use_df.to_parquet(f"./assets/data/{self.exchange_name}_use_df.prq")

        # Do univariate testing and record results
        if self.do_univariate_test:
            print("EXECUTE: Performing univariate testing")
            self.univariate_test(self.use_df)

        # Create initial model
        self.define_model()

        # Tune model
        if self.do_tune_models:
            self.tune_model()

        print("EXECUTE: Finished")
        return self.use_df

    def get_symbols_list(self):
        # Create list of symbols to loop through
        symbols = self.get_symbols()
        symbols = symbols[symbols['exchangeShortName'] == self.exchange_name].copy()
        self.symbols_list = symbols['symbol'].tolist()

    def get_data_dict(self, dictionary='price'):
        results_dict = {}

        if dictionary == 'price':
            # Query price data
            for symb in tqdm(self.symbols_list):
                df = self.get_price_data(symb)
                results_dict[symb] = df

            # Dump price dictionary
            with open(self.price_dict_location, 'wb') as f:
                pickle.dump(results_dict, f)
        elif dictionary == 'call':
            # Query call data
            for symb in tqdm(self.symbols_list):
                df = self.get_call_data(symb)
                results_dict[symb] = df

            # Dump call dictionary
            with open(self.call_dict_location, 'wb') as f:
                pickle.dump(results_dict, f)

        return results_dict

    def match_price_call_keys(self, price_dict, call_dict):
        price_dict = self.check_price_dictionary_data_quality(price_dict,
                                                              sparse_data_threshold=30,
                                                              repeat_data_threshold=5,
                                                              low_volume_threshold=1000)

        # Determine all keys
        price_keys = set(price_dict.keys())
        call_keys = set(call_dict.keys())

        # Determine keys that were cleaned from check dictionary data quality
        price_dropped_keys = list(call_keys - price_keys)

        # Determine which keys from call have blank dfs
        call_empty_keys = []
        for key, df in call_dict.items():
            if df.empty:
                call_empty_keys.append(key)

        # Create list of all keys needed to be dropped from call data
        call_drop_keys = set(price_dropped_keys + call_empty_keys)

        print(f"Dropped {len(call_drop_keys)} of {len(call_dict)} symbols from call data dictionary")
        # Drop bad keys from call data
        for key in call_drop_keys:
            del call_dict[key]
            try:
                del price_dict[key]
            except KeyError:
                continue

        return price_dict, call_dict


    def load_data_dict(self, dictionary='price'):
        if dictionary == 'price':
            with open(self.price_dict_location, 'rb') as f:
                data = pickle.load(f)

        if dictionary == 'call':
            with open(self.call_dict_location, 'rb') as f:
                data = pickle.load(f)

        return data

    def data_dict_to_dataframe(self, data):
        for key, df in data.items():
            df['symbol'] = key
            data[key] = df

        # Create concated df
        concat_df = pd.concat(data, ignore_index=True)
        return concat_df


    def engineer_price_data_features(self, df):
        # Group data by symbol
        grouped_df = df.groupby('symbol', group_keys=False)

        # Apply engineer features
        engineered_df = grouped_df.progress_apply(self.calc_engineer_features,
                                                  do_target_variable=self.do_target_variable,
                                                  do_percent_change=self.do_percent_change,
                                                  do_rolling_avg=self.do_rolling_avg,
                                                  do_lags=self.do_lags,
                                                  do_volatility=self.do_volatility,
                                                  do_relative_strength=self.do_relative_strength,
                                                  do_bollinger_bad=self.do_bollinger_bad,
                                                  do_macd=self.do_macd,
                                                  do_min_max_scale=self.do_min_max_scale,
                                                  target_periods=self.target_periods,
                                                  percent_change_periods=self.percent_change_periods,
                                                  rolling_avg_periods=self.rolling_avg_periods,
                                                  lags=self.lags,
                                                  volatility_windows=self.volatility_windows,
                                                  relative_strength_periods=self.relative_strength_periods,
                                                  bollinger_band_params=self.bollinger_band_params,
                                                  macd_parms=self.macd_params,
                                                  target_variable_col=self.target_variable_col,
                                                  percent_change_col=self.percent_change_col,
                                                  rolling_avg_col=self.rolling_avg_col,
                                                  lags_col=self.lags_col,
                                                  volatility_col=self.volatility_col,
                                                  relative_strength_col=self.relative_strength_col,
                                                  bollinger_band_col=self.bollinger_band_col,
                                                  macd_col=self.macd_col,
                                                  min_max_scale_col=self.min_max_scale_col)
        return engineered_df

    def add_index_features(self, df):
        # Get price data for VTI (potentially other index in future
        index = "VTI"
        index_df = self.get_price_data(index)
        index_df = index_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

        if self.do_engineer_index:
            print("engineer_index_features: do engineer index")
            # Apply feature engineering to Index df
            engineered_index = self.calc_engineer_features(index_df,
                                                           do_target_variable=False,
                                                           do_percent_change=self.do_percent_change,
                                                           do_rolling_avg=self.do_rolling_avg,
                                                           do_lags=self.do_lags,
                                                           do_volatility=self.do_volatility,
                                                           do_relative_strength=self.do_relative_strength,
                                                           do_bollinger_bad=self.do_bollinger_bad,
                                                           do_macd=self.do_macd,
                                                           do_min_max_scale=self.do_min_max_scale,
                                                           target_periods=self.target_periods,
                                                           percent_change_periods=self.percent_change_periods,
                                                           rolling_avg_periods=self.rolling_avg_periods,
                                                           lags=self.lags,
                                                           volatility_windows=self.volatility_windows,
                                                           relative_strength_periods=self.relative_strength_periods,
                                                           bollinger_band_params=self.bollinger_band_params,
                                                           macd_parms=self.macd_params,
                                                           target_variable_col=self.target_variable_col,
                                                           percent_change_col=self.percent_change_col,
                                                           rolling_avg_col=self.rolling_avg_col,
                                                           lags_col=self.lags_col,
                                                           volatility_col=self.volatility_col,
                                                           relative_strength_col=self.relative_strength_col,
                                                           bollinger_band_col=self.bollinger_band_col,
                                                           macd_col=self.macd_col,
                                                           min_max_scale_col=self.min_max_scale_col)
        else:
            print("engineer_index_features: DO NOT engineer index")
            engineered_index = index_df.copy()

        # add prefix to index df
        engineered_index = engineered_index.add_prefix(f"{index}_")

        # merge index and original df
        merged_df = pd.merge(df, engineered_index, left_on='date', right_on=f"{index}_date", how='left')
        merged_df = merged_df.drop(f"{index}_date", axis=1)

        if self.do_engineer_index:
            # Collect all columns that are the mutual calculations between stock and index
            index_cols = [col for col in merged_df.columns if col.startswith(f"{index}_")]
            original_cols = []

            for col in merged_df.columns:
                index_col = f"{index}_" + col
                if index_col in index_cols:
                    original_cols.append(col)

            # Subtract original columns from index equivalent
            merged_df = merged_df.assign(**{f"index_diff_{col1}": merged_df[col1] - merged_df[col2] for col1, col2 in
                                            zip(original_cols, index_cols)})
        return merged_df

    def combine_call_price_data(self, call_df, price_df):
        # Change dates to datetime format
        price_df['date'] = pd.to_datetime(price_df['date'])
        call_df['date'] = pd.to_datetime(call_df['date'])

        # Sort for merge_asof
        price_df = price_df.sort_values(['date', 'symbol'])
        call_df = call_df.sort_values(['date', 'symbol'])

        # Merge dataframes
        merged_df = pd.merge_asof(call_df, price_df, on='date', by='symbol', direction='backward')
        return merged_df

    def univariate_test(self, df):
        # Drop date column if it exists
        df.drop('date', axis=1, errors='ignore', inplace=True)
        df.dropna(inplace=True)

        # Create categorical target features
        df = self.calc_categorical_target_variables(df)

        print("Defining target variables and features:")
        target_variables = [col for col in df.columns if 'target' in col]
        features = [col for col in df.columns if col not in target_variables]
        categorical_targets = [col for col in df.columns if '_categ' in col]

        # Define models and metrics to be recorded
        print("Testing:")
        self.univariate_test_loop(df, categorical_targets, features)

    def define_model(self):
        self.model = DecisionTreeClassifier(max_depth=10)

    def tune_model(self):
        # Create train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

        param_grid = {'min_samples_split': Integer(2, 6)}

        opt = BayesSearchCV(self.model,
                            param_grid,
                            n_iter=5,
                            n_jobs=-1,
                            random_state=42,
                            verbose=10)

        # Train and tune model
        opt.fit(X_train, y_train)

        # Score model
        tuned_score = opt.score(X_test, y_test)
        print(f"Final tuned score: {tuned_score}")

        # Dump best model
        joblib.dump(opt.best_estimator_, "./assets/models/tuned_model.pkl")
        self.model = opt.best_estimator_

        return opt.best_estimator_


