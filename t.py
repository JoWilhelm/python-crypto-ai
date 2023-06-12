import pandas as pd
import numpy as np


main_df = pd.read_csv("HistoricalDataClassified_2016_2023_ov40_th005.csv")

print("sells:", (len(main_df[main_df["target"] == 0])/len(main_df)))
print("buys:", (len(main_df[main_df["target"] == 1])/len(main_df)))
print("holdss:", (len(main_df[main_df["target"] == 2])/len(main_df)))