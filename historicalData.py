from poloniex import Poloniex
import time
import pandas as pd
import numpy as np
from pandas import ExcelWriter

polo = Poloniex()


START = 1467331200 # 01.07.2016 dd.mm.yyyy
END = 1588284000 # 01.05.2020



# gets historical chart data from Poloniex API (300s candles)
def get_ChartData(coin, start, end):
    raw = polo.returnChartData(f"USDT_{coin}", 300, start, end)
    df = pd.DataFrame(raw)
    df.rename(columns={"close":f"{coin}_close", "open":f"{coin}_open", "low":f"{coin}_low", "high":f"{coin}_high", "quoteVolume":f"{coin}_volume", "weightedAverage":f"{coin}_average"}, inplace=True)
    df.set_index("date", inplace=True)
    # select columns to be used
    df = df[[f"{coin}_close", f"{coin}_low", f"{coin}_high", f"{coin}_volume", f"{coin}_average"]]
    return df

# can be used to combine historical data of multiple coins
def combine_dfs(list_dfs):
    df = pd.DataFrame()
    for list_df in list_dfs:
        if len(df) == 0:
            df = list_df
        else:
            df = df.join(list_df)
    return df
    



dataset = pd.DataFrame()

# collect data in 30d intervals (the API doesn't allow larger requests)
intervalStart = START
intervalEnd = START + 2592000 # +30d
monthsCounter = 1
while(intervalEnd < END):
    dataset = dataset.append(get_ChartData("BTC", intervalStart, intervalEnd), ignore_index=True)
    # shift interval 30d
    intervalStart = intervalEnd
    intervalEnd += 2592000 # +30d
    # counter
    print("months: ", monthsCounter)
    monthsCounter += 1
intervalEnd = END
dataset = dataset.append(get_ChartData("BTC", intervalStart, intervalEnd), ignore_index=True)
print("months: ", monthsCounter)
dataset = dataset.apply(pd.to_numeric)


# add additional columns
dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]


# print, to csv
print(dataset)
dataset.to_csv("HistoricalData.csv", index=False)