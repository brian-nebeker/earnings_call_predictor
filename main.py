import pandas as pd

df = pd.read_parquet('scrape.prq')

df['date'].value_counts()['MISSING']
df.columns