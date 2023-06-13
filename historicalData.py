from poloniex import Poloniex
import time
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import time

polo = Poloniex()


START = 1467331200 # 01.07.2016 dd.mm.yyyy
END = 1588284000 # 01.05.2020
END = 1680367000 # 01.04.2023

CANDLES_PERIOD = 21600


# gets historical chart data from Poloniex API (300s candles)
def get_ChartData(coin, start, end):
    raw = polo.returnChartData(f"USDT_{coin}", CANDLES_PERIOD, start, end)
    df = pd.DataFrame(raw)
    df.rename(columns={"date":"timestamp", "close":f"{coin}_close", "open":f"{coin}_open", "low":f"{coin}_low", "high":f"{coin}_high", "quoteVolume":f"{coin}_volume", "weightedAverage":f"{coin}_average"}, inplace=True)
    #df.set_index("date", inplace=True)
    # select columns to be used
    df = df[["timestamp", f"{coin}_close", f"{coin}_low", f"{coin}_high", f"{coin}_volume", f"{coin}_average"]]
    df["timestamp"] = df["timestamp"]/1000
    #print("len api response:", len(df))
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

intervalLength = 500*CANDLES_PERIOD
intervalStart = START
intervalEnd = START + intervalLength
intervalsCounter = 1
numIntervals = int((END - START) / intervalLength)


# for when API breaks access
## load DF
#intervalsCounter = 150
#dataset = pd.read_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv")
#intervalStart = START + (intervalLength * intervalsCounter)
#intervalEnd = intervalStart + intervalLength
#intervalsCounter += 1

while(intervalEnd < END):
    dataset = pd.concat([dataset, get_ChartData("BTC", intervalStart, intervalEnd)], ignore_index=True)
    # shift interval
    intervalStart = intervalEnd
    intervalEnd += intervalLength
    # counter
    print("intervals: ", intervalsCounter, "/", numIntervals, " len dataset: ", len(dataset))
    
    if intervalsCounter % 50 == 0:
        dataset = dataset.apply(pd.to_numeric)
        # add additional columns
        dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
        # print, to csv
        dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv", index=False)
        time.sleep(60)
    intervalsCounter += 1

intervalEnd = END
dataset = pd.concat([dataset, get_ChartData("BTC", intervalStart, intervalEnd)], ignore_index=True)


dataset = dataset.apply(pd.to_numeric)
# add additional columns
dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
# print, to csv
print(dataset)
dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}.csv", index=False)