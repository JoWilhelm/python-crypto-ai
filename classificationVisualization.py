import time
from poloniex import Poloniex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

polo = Poloniex()

def getPrices(coin):
    start = 1580515200 # 01.02.2020
    end = 1583798400 # 10.03.2020
    while True:
        try:
            raw = polo.returnChartData(f"USDT_{coin}", 300, start, end)
        except:
            print("connection lost, trying again")
            time.sleep(60)
            pass
        else:
            # connected
            break
    df = pd.DataFrame(raw)
    prices = df["close"].to_list()
    return prices


# looks |radius| candles ahead
def classifyFuture(prices, radius, threshhold=0):
    res = []
    for i in range(0, len(prices) - radius):
        pctChangeAvg = np.mean(prices[i+1:i+radius+1]) / prices[i];
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
    for i in range(radius, len(prices) - radius):
        pctChangeAvg = np.mean(prices[i-radius-1:i+radius+1]) / prices[i];
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
    for i in range(0, len(targetArrays[0])):
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
    #res = overlap([classifyFuture(prices, 20), classifyPastFuture(prices, 20)])
    res = classifyPastFuture(prices, 40, 0.0075)
    return res



# load price data
prices = getPrices("BTC")
prices = [float(price) for price in prices]
prices = [round(price, 2) for price in prices]

# get target labels
targets = classify(prices)

buyPrices = []
buyTimes = []
sellPrices = []
sellTimes = []
for i in range(0, len(prices)):
    if targets[i] == 1:
        # buy
        buyPrices.append(prices[i])
        buyTimes.append(i)
    if targets[i] == 0:
        # sell
        sellPrices.append(prices[i])
        sellTimes.append(i)


print("buys:", len(buyTimes), (len(buyTimes)/len(prices))*100, "% , sells:", len(sellTimes), (len(sellTimes)/len(prices))*100, "%")

plt.plot(prices)
plt.plot(buyTimes, buyPrices, 'go')
plt.plot(sellTimes, sellPrices, 'ro')
plt.show()

