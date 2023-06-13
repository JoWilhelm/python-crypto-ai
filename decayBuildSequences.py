from collections import deque
from xmlrpc.client import MAXINT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_300 = pd.read_csv("data/aligned/HistoricalData_2016_2023_300.csv")
df_900 = pd.read_csv("data/aligned/HistoricalData_2016_2023_900.csv")
df_3600 = pd.read_csv("data/aligned/HistoricalData_2016_2023_3600.csv")
df_21600 = pd.read_csv("data/aligned/HistoricalData_2016_2023_21600.csv")


df_300 = df_300[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_300["timescale"] = 300
df_900 = df_900[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_900["timescale"] = 900
df_3600 = df_3600[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_3600["timescale"] = 3600
df_21600 = df_21600[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_21600["timescale"] = 21600





# dfs should be ordered fine to coarse with the finest one having the labels
# seqLen should be divisible by the number of time-resolutions len(dfs)
def buildTimeDecaySequences(SEQLEN, dfs, timescales, CANDLES_SHIFT):
    numTimeScales = len(dfs)
    seqPartLen = int(SEQLEN/numTimeScales)

    seqDfs = []
    
    finestEndIndex = len(dfs[0])-1
    while True:
        
        print("finestEndIndex:", finestEndIndex)
        lastTimestamp = dfs[0]["timestamp"].iloc[finestEndIndex]+1
        seqDf = pd.DataFrame()
        for i in range(numTimeScales):

            # get prev index. Last with timestamp < lastTimestamp
            endIndex = dfs[i][dfs[i]["timestamp"] < lastTimestamp].index[-1]
            seqDf = pd.concat([dfs[i][endIndex-seqPartLen:endIndex], seqDf])
            lastTimestamp = seqDf["timestamp"].iloc[0]
        
        seqDfs.append(seqDf)
        finestEndIndex -= CANDLES_SHIFT

    return seqDfs   




    #res = []
#
    #while len(df) >= SEQ_LEN:
    #    first = df.head(SEQ_LEN).copy()
    #    first.index = np.arange(0, len(first))
    #    res.append(first)
    #    df = df.tail(len(df) - CANDLES_SHIFT)
    #    df.index = np.arange(0, len(df))
#
    #print("-done")
    #print("")
    #return res



#print(df_900)
#seqDf = buildTimeDecaySequences(80, [df_21600, df_3600, df_900, df_300], [21600, 3600, 900, 300])
seqDf = buildTimeDecaySequences(80, [df_300, df_900, df_3600, df_21600], [300, 900, 3600, 21600], 5000)
print(seqDf)

plt.plot(seqDf["timestamp"], seqDf["BTC_close"])
plt.show()