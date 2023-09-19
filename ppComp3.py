#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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



def splitDf_new(df):
    
    res = []
    print("")
    print("splitDf")
    while len(df) >= SEQ_LEN:
        first = df.head(SEQ_LEN).copy()
        first.index = np.arange(0, len(first))
        res.append(first)
        df = df.tail(len(df) - CANDLES_SHIFT)
        df.index = np.arange(0, len(df))

    print("-done")
    print("")
    return res



import numpy as np

def new_preprocess(df):

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








df = pd.read_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv")
df = df[['low', 'high', 'open', 'close', 'quantity_baseUnits', 'HLPercent']]

df = df.replace(0, 0.00001)

#df = df.tail(1000)
print(df)



splittedDfs = splitDf(df)
print(splittedDfs[0])

ppDfs = preprocess(splittedDfs)
print(ppDfs[0])

print(len(ppDfs))




#plt.plot(ppDfs[0]['close'])
#
#plt.show()




## Initialize the plot
#fig, ax = plt.subplots()
#
## Initialize a line plot. Note that the data for the line will be updated later.
#line, = ax.plot([], [], lw=2)
#
## Function to initialize the plot (required by FuncAnimation)
#def init():
#    line.set_data([], [])
#    return line,
#
## Function to update each frame of the plot
#def update(frame):
#    xdata = range(len(ppDfs[frame]['close']))
#    ydata = ppDfs[frame]['close']
#    ax.set_xlim(0, len(xdata) - 1)
#    ax.set_ylim(min(ydata) - 0.1, max(ydata) + 0.1)  # Optionally set y-limits to adjust automatically
#    line.set_data(xdata, ydata)
#    return line,
#
## Create an animation
#ani = FuncAnimation(
#    fig,
#    update,
#    frames=range(len(ppDfs)),  # Number of frames
#    init_func=init,
#    blit=True,
#    interval=10  # Show each frame for 0.5 seconds (500 milliseconds)
#)
#
#plt.show()


















