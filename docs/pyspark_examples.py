import pandas as pd

from pyspark.sql import functions as F, types as T, SparkSession

# initialize spark session
spark = SparkSession.builder.getOrCreate()

# read data
base_path = "/home/brett/git/earnings_call_predictor/docs"
df = pd.read_csv(f"{base_path}/sample_price_data.csv")
sdf = spark.read.csv(f"{base_path}/sample_price_data.csv",
                     header=True)

# spark is a lazy executor, so it doesn't actually run the process until you cache/some other "trigger" operation
sdf.cache().count()

# add column
df.loc[:, "price_vol"] = df.loc[:, "close_price"] + df.loc[:, "volume"]
sdf = sdf.withColumn("price_vol",
                     F.col("close_price") + F.col("volume"))
sdf.cache().count()

# rename column
df.rename({"price_vol": "pv"},
          axis=1,
          inplace=True)

sdf = sdf.withColumnRenamed("price_vol",
                            "pv")
sdf.cache().count()
