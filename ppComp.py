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


















