import pandas as pd

df = pd.read_parquet('nasdaq_price_100.prq')

df.describe()
df.info()