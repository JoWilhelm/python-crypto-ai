from collections import deque
from xmlrpc.client import MAXINT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation



df_30m_classified = pd.read_csv("historicalData/aligned/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_30_ov40_th015p.csv")
df_2h = pd.read_csv("historicalData/aligned/HistoricalData_BTC_USDT_01072016_01072023_HOUR_2.csv")
df_24h = pd.read_csv("historicalData/aligned/HistoricalData_BTC_USDT_01072016_01072023_DAY_1.csv")


#df_30m_classified = df_30m_classified[["ts_closeTime", "close", "BTC_volume", "BTC_HLPercent"]]
df_30m_classified["timescale"] = 0.5
#df_900 = df_900[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_2h["timescale"] = 2
#df_21600 = df_21600[["timestamp", "BTC_close", "BTC_volume", "BTC_HLPercent"]]
df_24h["timescale"] = 24

print(df_30m_classified.columns)



# composes dfs of individual sequences from different timescales
# dfs should be ordered fine to coarse with the finest one having the labels
# seqLen should be divisible by the number of time-resolutions len(dfs)
def composeTimeDecaySequences(SEQLEN, dfs, timescales, CANDLES_SHIFT):
    numTimeScales = len(dfs)
    seqPartLen = int(SEQLEN/numTimeScales)

    seqDfs = []
    finestEndIndex = len(dfs[0])-1
    numSequences = int((len(dfs[0]) - int(seqPartLen*(sum(timescales))/timescales[0]))/CANDLES_SHIFT)
    for _ in tqdm(range(numSequences)):
        
        #print("finesfinestEndIndex:",finestEndIndex)
        lastTimestamp = dfs[0]["ts_closeTime"].iloc[finestEndIndex]+1
        #print("lastTimestamp:", lastTimestamp)
        seqDf = pd.DataFrame()
        for i in range(numTimeScales):

            # get prev index. Last with ts_closeTime < lastTimestamp
            #print(dfs[i][dfs[i]["ts_closeTime"] < lastTimestamp])
            endIndex = dfs[i][dfs[i]["ts_closeTime"] < lastTimestamp].index[-1]
            seqDf = pd.concat([dfs[i][endIndex-seqPartLen:endIndex], seqDf])
            lastTimestamp = seqDf["ts_closeTime"].iloc[0]
        
        seqDfs.append(seqDf)
        finestEndIndex -= CANDLES_SHIFT

    return seqDfs[::-1]






seqDfs = composeTimeDecaySequences(90, [df_30m_classified, df_2h, df_24h], [1800, 7200, 86400], 2)


#print(seqDfs[0])
#
#plt.figure("first")
#plt.plot(seqDfs[0]["ts_closeTime"], seqDfs[0]["close"])
#
#plt.figure("last")
#plt.plot(seqDfs[-1]["ts_closeTime"], seqDfs[-1]["close"])
#
#plt.show()









## animate through all the sequences

fig, ax = plt.subplots()

# Initial plot
line, = ax.plot(seqDfs[0]["ts_closeTime"], seqDfs[0]["close"])
ax.set_title('close vs ts_closeTime')
ax.set_xlabel('ts_closeTime')
ax.set_ylabel('close')

# Update function for the animation
def update(num):
    ax.clear()  # Clear previous plot
    line, = ax.plot(seqDfs[num]["ts_closeTime"], seqDfs[num]["close"])
    ax.set_title('close vs ts_closeTime')
    ax.set_xlabel('ts_closeTime')
    ax.set_ylabel('close')
    return line,

# Create the animation, looping back at the end
ani = FuncAnimation(fig, update, frames=range(len(seqDfs)), interval=1, blit=True, repeat=True)

plt.show()