import time
import pickle
from tqdm import tqdm
from _mixins import NedHedgeFundMixins


class NedHedgeFund(NedHedgeFundMixins):
    def __init__(self):
        pass

    def thefuck(self):
        self.example()

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
            pickle.dump(temp, f)

        # Finsh with report for elapsed time for length of symbol list
        print(f"Created results dictionary. Elapsed time for {len(symbols_list)} symbols: {(time.time() - start_time) / 60:.1f}min")
        return results_dict

    def univ_testing(self):
        pass


n = NedHedgeFund()
temp = n.get_all_price_data_dict()


