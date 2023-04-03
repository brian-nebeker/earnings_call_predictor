import time
import pickle
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from _mixins import NedHedgeFundMixins
tqdm.pandas()


class NedHedgeFund(NedHedgeFundMixins):
    def __init__(self,
                 exchange_name="NYSE",
                 query_price_data: bool = False,
                 do_engineer_features: bool = False,
                 univariate_test: bool = False,
                 tune_models: bool = False,
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
        self.engineered_df = None
        self.price_df = None
        self.price_dict = None
        self.exchange_name = exchange_name
        self.query_price_data = query_price_data
        self.do_engineer_features = do_engineer_features
        self.univariate_test = univariate_test
        self.tune_models = tune_models
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

    def execute(self):
        # If true, generate all new price data and convert to vert stacked dataframe, if false load dict and convert
        if self.query_price_data:
            print("Querying new data")
            self.price_dict = self.get_price_data_dictionary()
            self.price_df = self.price_dict_to_dataframe(self.price_dict)
        else:
            print("Loading previous data")
            self.price_df = self.price_dict_to_dataframe()

        # If true, engineer features for stacked dataframe
        if self.do_engineer_features:
            self.engineered_df = self.engineer_price_data_features(self.price_df)

    def get_price_data_dictionary(self):
        # Create list of symbols to loop through
        symbols = self.get_symbols()
        symbols = symbols[symbols['exchangeShortName'] == self.exchange_name].copy()
        symbols_list = symbols['symbol'].tolist()

        # Loop through symbols list to create price data and store into a dictionary
        results_dict = {}
        for symb in tqdm(symbols_list):
            df = self.get_price_data(symb)
            results_dict[symb] = df

        # Dump dictionary
        with open(f'./assets/data/{self.exchange_name}_price_dictionary.pickle', 'wb') as f:
            pickle.dump(results_dict, f)

        # Finsh with report for elapsed time for length of symbol list
        print(f"Created results dictionary. Elapsed time for {len(symbols_list)} symbols: {(time.time() - start_time) / 60:.1f}min")
        return results_dict

    def price_dict_to_dataframe(self, data=None):
        if data is None:
            print("Loading dictionary data")
            with open(f'./assets/data/{self.exchange_name}_price_dictionary.pickle', 'rb') as f:
                data = pickle.load(f)

        # Add symbol column for group by
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
        print("Engineering features:")
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

        #  If do_add_index then add index columns and engineered index columns to
        print(f"do_add_index:{self.do_add_index}")
        print(f"do_engineer_index:{self.do_engineer_index}")
        if self.do_add_index:
            if self.do_engineer_index:
                print("Adding index with engineering")
                engineered_df = self.engineer_index_features(engineered_df)
            else:
                print("Adding index without engineering")
                index = "VTI"
                index_df = self.get_price_data(index)
                index_df = index_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                index_df = index_df.add_prefix(f"{index}_")

                engineered_df = pd.merge(engineered_df, index_df, left_on='date', right_on=f"{index}_date", how='left')
                engineered_df = engineered_df.drop(f"{index}_date", axis=1)
        return engineered_df

    def engineer_index_features(self, df):
        # Get price data for VTI (potentially other index in future
        print("Query Index")
        index = "VTI"
        index_df = self.get_price_data(index)
        index_df = index_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

        print("Apply feature engineering to index df")
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

        print("Add index prefix")
        # add prefix to index df
        engineered_index = engineered_index.add_prefix(f"{index}_")

        print("Merge index prefix")
        # merge index and original df
        merged_df = pd.merge(df, engineered_index, left_on='date', right_on=f"{index}_date", how='left')
        merged_df = merged_df.drop(f"{index}_date", axis=1)

        index_cols = [col for col in merged_df.columns if col.startswith(f"{index}_")]
        original_cols = []

        for col in merged_df.columns:
            index_col = f"{index}_" + col
            if index_col in index_cols:
                original_cols.append(col)

        print("Subtract index columns from original using Assign")
        # Subtract original columns from index equivalent
        merged_df = merged_df.assign(**{f"index_diff_{col1[4:]}": merged_df[col1] - merged_df[col2] for col1, col2 in
                                        zip(original_cols, index_cols)})
        return merged_df

    def univ_testing(self):
        print(f"load_new_data: {self.query_price_data}")
        print(f"engineer_features: {self.do_engineer_features}")

        if self.query_price_data:
            # TODO: Create load_new_data option
            print("NO CODE FOR QUERY PRICE DATA TRUE")
            return
        else:
            concat_df = self.price_dict_to_dataframe()

        if self.do_engineer_features:
            engineered_df = self.engineer_price_data_features(concat_df)

            # Drop columns
            drop_columns = ['label', 'symbol', 'date', 'adjClose']
            engineered_df = engineered_df.drop(drop_columns, axis=1)

            # Drop NAs
            # TODO: switch to checking for 252+ days of data, validate by creating a pandas daterange for trading days
            engineered_df.dropna(inplace=True)
            engineered_df.to_parquet('./assets/data/stock_price_data_engineered.prq')
        else:
            engineered_df = pd.read_parquet('./assets/data/stock_price_data_engineered.prq')

        # Define target variables and features
        print("Defining target variables and features:")
        target_variables = [col for col in engineered_df.columns if 'target' in col]
        features = [col for col in engineered_df.columns if col not in target_variables]

        # Create categorical target features
        for col in target_variables:
            temp = pd.qcut(engineered_df[col], 10, labels=False).copy()
            temp[temp <= 8] = 0
            temp[temp > 8] = 1
            engineered_df[col + '_categ'] = temp

        categorical_variables = [col for col in engineered_df.columns if '_categ' in col]

        # Define models and metrics to be recorded
        print("Defining models:")
        logit = LogisticRegression()
        tree = DecisionTreeClassifier(max_depth=5)
        tested_column = []
        tested_target = []
        tested_accuracy_score_logistic = []
        tested_f1_score_logistic = []
        tested_accuracy_score_tree = []
        tested_f1_score_tree = []

        # Loop every for every categorical target
        for y_col in categorical_variables:
            print(f"Testing for {y_col}")
            train, test = train_test_split(engineered_df, train_size=0.33, random_state=42, stratify=engineered_df[y_col])
            for x_col in tqdm(features):
                # Record tests
                tested_column.append(x_col)
                tested_target.append(y_col)

                # Define train test split
                X_train = train[x_col].values.reshape(-1, 1).copy()
                X_test = test[x_col].values.reshape(-1, 1).copy()
                y_train = train[y_col].values.reshape(-1, 1).copy()
                y_test = test[y_col].values.reshape(-1, 1).copy()

                # Fit logit models
                try:
                    logit.fit(X_train, y_train.ravel())
                    logit_pred = logit.predict(X_test)
                    tested_accuracy_score_logistic.append(accuracy_score(y_test, logit_pred))
                    tested_f1_score_logistic.append(f1_score(y_test, logit_pred))
                except ValueError:
                    tested_accuracy_score_logistic.append("ValueError")
                    tested_f1_score_logistic.append("ValueError")

                # Fit tree models
                try:
                    tree.fit(X_train, y_train.ravel())
                    tree_pred = tree.predict(X_test)
                    tested_accuracy_score_tree.append(accuracy_score(y_test, tree_pred))
                    tested_f1_score_tree.append(f1_score(y_test, tree_pred))
                except ValueError:
                    tested_accuracy_score_tree.append("ValueError")
                    tested_f1_score_tree.append("ValueError")

        results = {"column": tested_column,
                   "target": tested_target,
                   "accuracy_score_logistic": tested_accuracy_score_logistic,
                   "f1_score_logistic": tested_f1_score_logistic,
                   "accuracy_score_tree": tested_accuracy_score_tree,
                   "f1_score_tree": tested_f1_score_tree
                   }

        # Record results
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        experimentation_results = pd.DataFrame(results)
        experimentation_results.to_csv(f"./assets/experiment_results/experimentation_results_{datetime_str}.csv")

