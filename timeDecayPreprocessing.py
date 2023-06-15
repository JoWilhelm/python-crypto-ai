import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tqdm import tqdm



means_300 = {"BTC_close": 0.000000691, 
             "BTC_low": 0.000000685, 
             "BTC_high": 0.000000677, 
             "BTC_average": 0.000000678, 
             "BTC_volume": 9.30, 
             "BTC_HLPercent": 0.00269}
stds_300 = {"BTC_close": 0.000318, 
             "BTC_low": 0.000298, 
             "BTC_high": 0.000273, 
             "BTC_average": 0.000275, 
             "BTC_volume": 2.815, 
             "BTC_HLPercent": 0.003189}


means_900 = {"BTC_close": 0.000002054, 
             "BTC_low": 0.000002047, 
             "BTC_high": 0.000002019, 
             "BTC_average": 0.000002018, 
             "BTC_volume": 10.88, 
             "BTC_HLPercent": 0.00503}
stds_900 = {"BTC_close": 0.000520, 
             "BTC_low": 0.000504, 
             "BTC_high": 0.000449, 
             "BTC_average": 0.000448, 
             "BTC_volume": 2.209, 
             "BTC_HLPercent": 0.00534}


means_3600 = {"BTC_close": 0.0000081421, 
             "BTC_low": 0.000008189, 
             "BTC_high": 0.000007983, 
             "BTC_average": 0.0000081421, 
             "BTC_volume": 10.88, 
             "BTC_HLPercent": 0.00503}
stds_3600 = {"BTC_close": 0.000520, 
             "BTC_low": 0.000504, 
             "BTC_high": 0.000449, 
             "BTC_average": 0.000448, 
             "BTC_volume": 2.209, 
             "BTC_HLPercent": 0.00534}


def preprocess(df):
    #df = df.replace([0.0], 0.0001)

    for col in ["BTC_close", "BTC_low", "BTC_high", "BTC_average"]:
        df[col] = np.log(df[col])
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        mean = np.mean(df[col])
        #mean = 0.00000068
        std = np.std(df[col])
        #std = 0.00028
        print(col)
        print("mean:", mean, ", std:", std)
        df[col] = (df[col] - mean) / std
    
    
    df["BTC_volume"] = df["BTC_volume"].replace(0, 1)
    df["BTC_volume"] = np.log(df["BTC_volume"])
    #df["BTC_volume"] = df["BTC_volume"].pct_change()   # taking the pct change somehow makes it worse
    #df.dropna(inplace=True)                            # taking the pct change somehow makes it worse
    mean = np.mean(df["BTC_volume"])
    #mean = 9.3
    std = np.std(df["BTC_volume"])
    #std = 2.82
    print("BTC_volume")
    print("mean:", mean, ", std:", std)
    df["BTC_volume"] = (df["BTC_volume"] - mean) / std


    mean = np.mean(df["BTC_HLPercent"])
    #mean = 0.0027
    std = np.std(df["BTC_HLPercent"])
    #std = 0.0032
    print("BTC_HLPercent")
    print("mean:", mean, ", std:", std)
    df["BTC_HLPercent"] = (df["BTC_HLPercent"] - mean) / std


    return df




df_300_classified = pd.read_csv("data/aligned/HistoricalDataClassified_2016_2023_300_ov40_th04.csv")
df_900 = pd.read_csv("data/aligned/HistoricalData_2016_2023_900.csv")
df_3600 = pd.read_csv("data/aligned/HistoricalData_2016_2023_3600.csv")
df_21600 = pd.read_csv("data/aligned/HistoricalData_2016_2023_21600.csv")


df_300_classified["timescale"] = 5
df_900["timescale"] = 15
df_3600["timescale"] = 60
df_21600["timescale"] = 360

print(df_3600)
df_3600_pp = preprocess(df_3600)
print(df_3600_pp)






