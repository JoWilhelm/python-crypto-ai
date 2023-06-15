from collections import deque
from xmlrpc.client import MAXINT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation



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




# composes dfs of individual sequences from different timescales (in reverse)
# dfs should be ordered fine to coarse with the finest one having the labels
# seqLen should be divisible by the number of time-resolutions len(dfs)
def composeTimeDecaySequences(SEQLEN, dfs, timescales, CANDLES_SHIFT):
    numTimeScales = len(dfs)
    seqPartLen = int(SEQLEN/numTimeScales)

    seqDfs = []
    finestEndIndex = len(dfs[0])-1
    for _ in tqdm(range(int((len(dfs[0]) -int(seqPartLen*(sum(timescales))/300))/CANDLES_SHIFT))): # calculating how many sequences there will be
        
        lastTimestamp = dfs[0]["timestamp"].iloc[finestEndIndex]+1
        seqDf = pd.DataFrame()
        for i in range(numTimeScales):

            # get prev index. Last with timestamp < lastTimestamp
            endIndex = dfs[i][dfs[i]["timestamp"] < lastTimestamp].index[-1]
            seqDf = pd.concat([dfs[i][endIndex-seqPartLen:endIndex], seqDf])
            lastTimestamp = seqDf["timestamp"].iloc[0]
        
        seqDfs.append(seqDf)
        finestEndIndex -= CANDLES_SHIFT

    return seqDfs[::-1]






seqDfs = composeTimeDecaySequences(80, [df_300, df_900, df_3600, df_21600], [300, 900, 3600, 21600], 10)

#plt.figure("first")
#plt.plot(seqDfs[0]["timestamp"], seqDfs[0]["BTC_close"])
#
#plt.figure("last")
#plt.plot(seqDfs[-1]["timestamp"], seqDfs[-1]["BTC_close"])
#
#plt.show()









## animate through all the sequrnces

fig, ax = plt.subplots()

# Initial plot
line, = ax.plot(seqDfs[0]["timestamp"], seqDfs[0]["BTC_close"])
ax.set_title('BTC_close vs timestamp')
ax.set_xlabel('timestamp')
ax.set_ylabel('BTC_close')

# Update function for the animation
def update(num):
    ax.clear()  # Clear previous plot
    line, = ax.plot(seqDfs[num]["timestamp"], seqDfs[num]["BTC_close"])
    ax.set_title('BTC_close vs timestamp')
    ax.set_xlabel('timestamp')
    ax.set_ylabel('BTC_close')
    return line,

# Create the animation, looping back at the end
ani = FuncAnimation(fig, update, frames=range(len(seqDfs)), interval=1, blit=True, repeat=True)

plt.show()