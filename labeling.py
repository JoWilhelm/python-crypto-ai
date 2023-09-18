import pandas as pd
import numpy as np
from tqdm import tqdm

historicalDataPath = "historicalData/HistoricalData_BTC_USDT_01072016_01072023_MINUTE_5.csv"

# looks |radius| candles ahead
def classifyFuture(prices, radius, threshhold=0):
    res = []
    for i in tqdm(range(0, len(prices) - radius)):
        pctChangeAvg = np.mean(prices[i+1:i+radius+1]) / prices[i]
        if pctChangeAvg >= 1 + threshhold:
            # future average price higher than current price, buy
            res.append(1)
        elif pctChangeAvg < 1 - threshhold:
            # future average price lower than current price, sell
            res.append(0)
        else:
            # price has not moved much, hold
            res.append(2)           
    # fill end with zeros
    for i in range(0, radius):
            res.append(0)
    return res

# looks |radius| candles back and ahead
def classifyPastFuture(prices, radius, threshhold=0):
    res = []
    # fill start with zeros
    for i in range(0, radius):
        res.append(0)
    for i in tqdm(range(radius, len(prices) - radius)):
        pctChangeAvg = np.mean(prices[i-radius-1:i+radius+1]) / prices[i]
        if pctChangeAvg >= 1 + threshhold:
            # average price higher than current price, buy
            res.append(1)
        elif pctChangeAvg < 1 - threshhold:
            # average price lower than current price, sell
            res.append(0)
        else:
            # price has not moved much, hold
            res.append(2)           
    # fill end with zeros
    for i in range(0, radius):
            res.append(0)
    return res

# target only 1 or 0 if all agree, else 2 (hold)
def overlap(targetArrays):
    res = []
    for i in tqdm(range(0, len(targetArrays[0]))):
        allEaqual = True
        for j in range(1, len(targetArrays)):
            if targetArrays[j][i] != targetArrays[0][i]:
                allEaqual = False
        if allEaqual:
            res.append(targetArrays[0][i])
        else:
            res.append(2)
    return res

def convertToActionOrHold(targets):
    return [0 if t==2 else 1 for t in targets]

# build classification strategy here
def classify(prices):
    res = overlap([classifyFuture(prices, 40, 0.004), classifyPastFuture(prices, 40, 0.004)])
    return res



# load DF
print("loading df...")
main_df = pd.read_csv(historicalDataPath)
print("done")
main_df.index = np.arange(0, len(main_df))
#main_df = main_df.replace([0.0], 0.0001)

# classify every row
print("classifying...")
main_df["target"] = classify(main_df[f"close"])

# to csv
main_df.to_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv", index=False)
