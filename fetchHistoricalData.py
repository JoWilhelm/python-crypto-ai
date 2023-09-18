import sys
sys.path.append('./exchangeInterface')
import marketData
import pandas as pd


pairing = 'BTC_USDT'
start = 1467374400 # 01.07.2016
end = 1688191200 # 01.07.2023
candleIntv = 'MINUTE_5'

dataset = marketData.getHistoricalData('BTC_USDT', start=start, end=end, candleIntv='MINUTE_5')
dataset["HLPercent"] = (dataset["high"] - dataset["low"]) / dataset["high"]

# print, to csv
print(dataset)
dataset.to_csv(f"historicalData/HistoricalData_{pairing}_01072016_01072023_{candleIntv}.csv", index=False)

