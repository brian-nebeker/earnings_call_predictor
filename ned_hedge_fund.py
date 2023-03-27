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
    def __init__(self):
        # TODO: add initiation variables
        pass

    def get_all_price_data_dict(self):
        # Initiate start time
        start_time = time.time()

        # Create list of symbols to loop through
        symbols = self.get_symbols()
        symbols = symbols[symbols['exchangeShortName'] == 'NYSE'].copy()
        symbols_list = symbols['symbol'].tolist()
        print(f"Created symbols. Elapsed: {time.time() - start_time:.1f}sec")

        # Loop through symbols list to create price data and store into a dictionary
        results_dict = {}
        for symb in tqdm(symbols_list):
            df = self.get_price_data(symb)
            results_dict[symb] = df

        # Dump dictionary
        with open('./assets/data/stock_price_dictionary.pickle', 'wb') as f:
            pickle.dump(results_dict, f)

        # Finsh with report for elapsed time for length of symbol list
        print(f"Created results dictionary. Elapsed time for {len(symbols_list)} symbols: {(time.time() - start_time) / 60:.1f}min")
        return results_dict

    # TODO: create engineer_features function

    def univ_testing(self, load_new_data=False, engineer_features=False):
        print(f"load_new_data: {load_new_data}")
        print(f"engineer_features: {engineer_features}")

        # TODO: Create load_new_data option

        if engineer_features:
            # Load pickled dictionary of stock data
            print("Loading dictionary data:")
            with open('./assets/data/stock_price_dictionary.pickle', 'rb') as f:
                data = pickle.load(f)

            # Add symbol column for group by
            for key, df in data.items():
                df['symbol'] = key
                data[key] = df

            # Create concated df
            concat_df = pd.concat(data, ignore_index=True)

            # Create grouped by df
            grouped_df = concat_df.groupby('symbol', group_keys=False)

            # Define periods and windows for engineer feature
            target_periods = [1, 30, 60, 90, 252]
            percent_change_periods = [1, 30, 60, 90, 252]
            rolling_avg_periods = [10, 20, 30, 60, 120, 252]
            lags = [10, 20, 30, 60, 120, 252]
            vol_windows = [30, 90, 180, 252]
            rs_periods = [7, 14, 30]

            print("Engineering features:")
            engineered_df = grouped_df.progress_apply(self.engineer_features2,
                                                      target_periods=target_periods,
                                                      percent_change_periods=percent_change_periods,
                                                      rolling_avg_periods=rolling_avg_periods,
                                                      lags=lags, vol_windows=vol_windows,
                                                      rs_periods=rs_periods)

            # Drop columns
            drop_columns = ['label', 'symbol', 'date', 'adjClose']
            engineered_df = engineered_df.drop(drop_columns, axis=1)

            # Drop NAs
            engineered_df.dropna(inplace=True) # TODO: switch to checking for 252+ days of data, validate by creating a pandas daterange for stock trading days
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

