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



def preprocess1(df):
    # before sequencing

    # pct.change transform price columns ('low', 'high', 'open', 'close')
    # scale every colum (center mean and unit variance)

    for col in df.columns:
        if col != 'target':
            if col != 'quantity_baseUnits' and col != 'HLPercent':
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)

            df[col] = preprocessing.scale(df[col].values)
    df.index = np.arange(0, len(df))
    return df





def splitDf_after(df):

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








df = pd.read_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv")
df = df[['low', 'high', 'open', 'close', 'quantity_baseUnits', 'HLPercent']]
#df = df.tail(10000)
print(df)


ppDf = preprocess1(df)
#plt.plot(ppDf['quantity_baseUnits'])
#plt.show()


# Create subplots in a vertical layout
fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(5, 15))

# If there's only one subplot, axes will not be an array; convert it to an array for consistency
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

for i, column in enumerate(df.columns):
    df[column].plot(ax=axes[i])
    axes[i].set_title(column)

plt.tight_layout()
plt.show()










#splittedDfsPP = splitDf_after(ppDf)




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


















