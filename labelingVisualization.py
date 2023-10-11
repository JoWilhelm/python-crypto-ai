import sys
sys.path.append('./exchangeInterface')


import marketData

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go










START = 1580515200 # 01.02.2020
END = 	1585692000 # 01.04.2020
candleIntv = 'MINUTE_30'








# looks |radius| candles ahead
def classifyFuture(prices, radius, threshhold=0):
    res = []
    for i in range(0, len(prices) - radius):
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
    for i in range(radius, len(prices) - radius):
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
    res = overlap([classifyFuture(prices, 40, 0.0015), classifyPastFuture(prices, 40, 0.0015)])

    #res = overlap([classifyFuture(prices, 200, 0.0015), classifyPastFuture(prices, 200, 0.0015)])
    
    #res = overlap([classifyFuture(prices, 40, 0.0004), classifyPastFuture(prices, 40, 0.0004)])
    return res






# load price data
df = marketData.getHistoricalData('BTC_USDT', start=START, end=END, candleIntv=candleIntv)
print(df)






# get target labels
targets = classify(df['close'])

buyPrices = []
buyTimes = []
sellPrices = []
sellTimes = []
for i in range(0, len(df['close'])):
    if targets[i] == 1:
        # buy
        buyPrices.append(df['close'][i])
        buyTimes.append(df['dt_closeTime'][i])
    if targets[i] == 0:
        # sell
        sellPrices.append(df['close'][i])
        sellTimes.append(df['dt_closeTime'][i])


print("buys:", len(buyTimes), (len(buyTimes)/len(df['close']))*100, "% , sells:", len(sellTimes), (len(sellTimes)/len(df['close']))*100, "%")





# plot
fig = go.Figure(data=[go.Candlestick(x=df['dt_closeTime'],
                       open=df['open'], high=df['high'],
                       low=df['close'], close=df['close'],
                       name=f'BTC/USDT {candleIntv} candles')])

# Scatter plot overlay for green dots
fig.add_trace(go.Scatter(x=buyTimes,
                         y=buyPrices,
                         mode='markers',
                         marker=dict(color='green'),
                         name='labeled \'buy\''))

# Scatter plot overlay for red dots
fig.add_trace(go.Scatter(x=sellTimes,
                         y=sellPrices,
                         mode='markers',
                         marker=dict(color='red'),
                         name='labeled \'sell\''))

fig.update_yaxes(fixedrange=False)
fig.show()
