import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt



#main_df = pd.read_csv("data/HistoricalDataClassified_2016_2023_ov40_th04.csv")
#
#print("sells:", (len(main_df[main_df["target"] == 0])/len(main_df)))
#print("buys:", (len(main_df[main_df["target"] == 1])/len(main_df)))
#print("holdss:", (len(main_df[main_df["target"] == 2])/len(main_df)))


#df_300 = pd.read_csv("data/aligned2/HistoricalData_2016_2023_300.csv")
#df_900 = pd.read_csv("data/aligned2/HistoricalData_2016_2023_900.csv")
#df_3600 = pd.read_csv("data/aligned2/HistoricalData_2016_2023_3600.csv")
#df_21600 = pd.read_csv("data/aligned2/HistoricalData_2016_2023_21600.csv")
#
#plt.plot(df_300["timestamp"], df_300["BTC_close"])# *3  72 
#plt.plot(df_900["timestamp"], df_900["BTC_close"]) # *4 24
#plt.plot(df_3600["timestamp"], df_3600["BTC_close"]) # *6
#plt.plot(df_21600["timestamp"], df_21600["BTC_close"])
#plt.show()

arr = np.concatenate([np.full((15), 5), np.full((15), 15), np.full((15), 60), np.full((15), 360)])
std = np.std(arr)
print(std)