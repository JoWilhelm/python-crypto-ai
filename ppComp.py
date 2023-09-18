#import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np



SEQ_LEN = 180 #240   # how many past candles to use to predict
CANDLES_SHIFT = 2 #5 # how many candles to shift between sequences
NAME = "m5_ov40th04p_shift2_seq180"
VALIDATION_PCT = 0.2




def preprocess(dfs):
    for df in dfs:
        for col in df.columns:
            if col != "target":
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)
                df[col] = preprocessing.scale(df[col].values)
                df.index = np.arange(0, len(df))

    return dfs


def splitDf(df):
    res = []
    print("")
    print("splitDf")
    while len(df) >= SEQ_LEN + len(df.columns) -1:
        first = df.head(SEQ_LEN + len(df.columns) -1).copy()
        first.index = np.arange(0, len(first))
        res.append(first)
        df = df.tail(len(df) - CANDLES_SHIFT)
        df.index = np.arange(0, len(df))

    print("-done")
    print("")
    return res






df = pd.read_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv")
df = df[['low', 'high', 'open', 'close', 'quantity_baseUnits', 'HLPercent', 'dt_closeTime']]


df = df.tail(5000)
print(df)



splittedDfs = splitDf(df)

print(splittedDfs)


#plt.plot(df['close'])
#
#plt.show()























