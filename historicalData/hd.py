from ..exchangeInterface.marketData import getHistoricalData


pairing = 'BTC_USDT'
start = 1467374400 # 01.07.2016
end = 1688191200 # 01.07.202
candleIntv = 'MINUTE_5'



#dataset = getHistoricalData(pairing=pairing, start=start, end=end, candleIntv=candleIntv)

dataset = getHistoricalData('BTC_USDT', start=1694264564-(60*5*1600), end=1694264564, candleIntv='MINUTE_5')




dataset["HLPercent"] = (dataset["high"] - dataset["low"]) / dataset["high"]
# print, to csv
print(dataset)
dataset.to_csv(f"HistoricalData_{pairing}_01072016_01072023_{candleIntv}.csv", index=False)

