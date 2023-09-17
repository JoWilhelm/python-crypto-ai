import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tqdm import tqdm


historicalDataPath = "HistoricalData_2016_2023.csv"

# load DF
main_df = pd.read_csv(historicalDataPath)
main_df.index = np.arange(0, len(main_df))



def old_preprocess(df):
    df = df.replace([0.0], 0.0001)
    for col in df.columns:
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values)
        df.index = np.arange(0, len(df))

    return df



def new_preprocess(df):
    #df = df.replace([0.0], 0.0001)

    for col in ["BTC_close", "BTC_low", "BTC_high", "BTC_average"]:
        #print(col)
        df[col] = np.log(df[col])
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        #mean = np.mean(df[col])
        mean = 0.00000068
        #std = np.std(df[col])
        std = 0.00028
        #print("mean:", mean, ", std:", std)
        df[col] = (df[col] - mean) / std
    
    
    df["BTC_volume"] = df["BTC_volume"].replace(0, 1)
    df["BTC_volume"] = np.log(df["BTC_volume"])
    #df["BTC_volume"] = df["BTC_volume"].pct_change()   # taking the pct change somehow makes it worse
    #df.dropna(inplace=True)                            # taking the pct change somehow makes it worse
    #mean = np.mean(df["BTC_volume"])
    mean = 9.3
    #std = np.std(df["BTC_volume"])
    std = 2.82
    #print("mean:", mean, ", std:", std)
    df["BTC_volume"] = (df["BTC_volume"] - mean) / std


    #mean = np.mean(df["BTC_HLPercent"])
    mean = 0.0027
    #std = np.std(df["BTC_HLPercent"])
    std = 0.0032
    #print("mean:", mean, ", std:", std)
    df["BTC_HLPercent"] = (df["BTC_HLPercent"] - mean) / std


    return df





def rollingWindow(data, size=250):
    averages = []
    variances = []

    start = 0
    end = size-1
    for i in tqdm(range(int(len(data)/size)-1)):
        avg = np.mean([data[start:end]])
        var = np.var(data[start:end])
        averages.append(avg)
        variances.append(var)
        start = end
        end += size

    return averages, variances






df_oldPP = old_preprocess(main_df)
df_newPP = new_preprocess(main_df)


### PREPROCESSED DATA

## price
#plt.figure("price_old")
#plt.plot(df_oldPP["BTC_close"])
#plt.figure("price_new")
#plt.plot(df_newPP["BTC_close"])

## volume
#plt.figure("volume_old")
#plt.plot(df_oldPP["BTC_volume"])
#plt.figure("volume_new")
#plt.plot(df_newPP["BTC_volume"])

## HLPercent
#plt.figure("HLPercent_old")
#plt.plot(df_oldPP["BTC_HLPercent"])
#plt.figure("HLPercent_new")
#plt.plot(df_newPP["BTC_HLPercent"])



### ROLLING WINDOWS AVG VAR
#
## price
#avg_price_old, var_price_old = rollingWindow(df_oldPP["BTC_close"])
#avg_price_new, var_price_new = rollingWindow(df_newPP["BTC_close"])
#plt.figure("price_avg")
#plt.plot(avg_price_old)
#plt.plot(avg_price_new)
#plt.figure("price_var")
#plt.plot(var_price_old)
#plt.plot(var_price_new)
#
## volume
#avg_volume_old, var_volume_old = rollingWindow(df_oldPP["BTC_volume"])
#avg_volume_new, var_volume_new = rollingWindow(df_newPP["BTC_volume"])
#plt.figure("volume_avg")
#plt.plot(avg_volume_old)
#plt.plot(avg_volume_new)
#plt.figure("volume_var")
#plt.plot(var_volume_old)
#plt.plot(var_volume_new)
#
## HLPercent
#avg_HLPercent_old, var_HLPercent_old = rollingWindow(df_oldPP["BTC_HLPercent"])
#avg_HLPercent_new, var_HLPercent_new = rollingWindow(df_newPP["BTC_HLPercent"])
#plt.figure("HLPercent_avg")
#plt.plot(avg_HLPercent_old)
#plt.plot(avg_HLPercent_new)
#plt.figure("HLPercent_var")
#plt.plot(var_HLPercent_old)
#plt.plot(var_HLPercent_new)




plt.show()