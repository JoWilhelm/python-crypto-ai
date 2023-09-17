from polosdk import RestClient
import time
import pandas as pd
from pandas import ExcelWriter
from datetime import datetime

client = RestClient()


candleIntvSeconds = {
    'MINUTE_1': 60,
    'MINUTE_5': 60*5,
    'MINUTE_10': 60*10,
    'MINUTE_15': 60*15,
    'MINUTE_30': 60*30,
    'HOUR_1': 60*60,
    'HOUR_2': 60*60*2,
    'HOUR_4': 60*60*4,
    'HOUR_6': 60*60*6,
    'HOUR_12': 60*60*12,
    'DAY_1': 60*60*24,
    'DAY_3': 60*60*24*3,
    'WEEK_1': 60*60*24*7
}





def getHistoricalDataSingleRequest(pairing, start, end, candleIntv):
    """Makes a single API request to fetch historical market data. Limit 500 candles

    Args:
        pairing (String): e.g. 'BTC_USDT'
        start (int): timestamp start of interval
        end (int): timestamp end of interval
        candleIntv (String): e.g. 'MINUTE_5'
    Returns:
        pd.DataFrame 
        columns = ['low', 'high', 'open', 'close', 'amount_quoteUnits', 'quantity_baseUnits', 'buyTakerAmount_quoteUnits', 'buyTakerQuantity_baseUnits', 'tradeCount', 'ts_recordPushed', 'weightedAverage', 'ts_startTime', 'ts_closeTime', 'dt_close']
    """
    # poloniex API request
    response = client.markets().get_candles(pairing, candleIntv, start_time=start*1000, end_time=end*1000, limit=min(500, int((end-start)/candleIntvSeconds[candleIntv])))
    
    df = pd.DataFrame(response, columns=['low', 'high', 'open', 'close', 'amount_quoteUnits', 'quantity_baseUnits', 'buyTakerAmount_quoteUnits', 'buyTakerQuantity_baseUnits', 'tradeCount', 'ts_recordPushed', 'weightedAverage', 'interval', 'ts_openTime', 'ts_closeTime'])
    df.drop('interval', axis=1, inplace=True)
    # convert timestamps from milliseconds to seconds 
    df['ts_openTime'] = df['ts_openTime'] / 1000
    df['ts_closeTime'] = df['ts_closeTime'] / 1000
    df['ts_recordPushed'] = df['ts_recordPushed'] / 1000

    df = df.apply(pd.to_numeric)
    
    df['dt_closeTime'] = [datetime.fromtimestamp(ts) for ts in df['ts_closeTime']]
    

    return df


#df = getHistoricalDataSingleRequest('BTC_USDT', start=1583798400, end=1583798400+(60*5*100), candleIntv='MINUTE_5')
#print(df)




def getHistoricalData(pairing, start, end, candleIntv):
    """Breaks one long interval up into multiple API request in order to fetch more than 500 candles of historical market data.

    Args:
        pairing (String): e.g. 'BTC_USDT'
        start (int): timestamp start of interval
        end (int): timestamp end of interval
        candleIntv (String): e.g. 'MINUTE_5'
    Returns:
        pd.DataFrame 
        columns = ['low', 'high', 'open', 'close', 'amount_quoteUnits', 'quantity_baseUnits', 'buyTakerAmount_quoteUnits', 'buyTakerQuantity_baseUnits', 'tradeCount', 'ts_recordPushed', 'weightedAverage', 'ts_startTime', 'ts_closeTime', 'dt_close']
    """
    
    intervalLength = 500*candleIntvSeconds[candleIntv]
    intervalStart = start
    intervalEnd = start + intervalLength
    intervalsCounter = 1
    numIntervals = int((end - start) / intervalLength)

    df = pd.DataFrame()

    # for when API breaks access
    ## load DF
    #intervalsCounter = 150
    #dataset = pd.read_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv")
    #intervalStart = START + (intervalLength * intervalsCounter)
    #intervalEnd = intervalStart + intervalLength
    #intervalsCounter += 1

    while(intervalEnd < end):
        df = pd.concat([df, getHistoricalDataSingleRequest(pairing, intervalStart, intervalEnd, candleIntv)], ignore_index=True)
        # shift interval
        intervalStart = intervalEnd
        intervalEnd += intervalLength

        # counter
        print("intervals: ", intervalsCounter, "/", numIntervals, " len dataset: ", len(df))

        #if intervalsCounter % 50 == 0:
        #    dataset = dataset.apply(pd.to_numeric)
        #    # add additional columns
        #    dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
        #    # print, to csv
        #    dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv", index=False)
        #    time.sleep(60)
        intervalsCounter += 1

    intervalEnd = end
    df = pd.concat([df, getHistoricalDataSingleRequest(pairing, intervalStart, intervalEnd, candleIntv)], ignore_index=True)

    # add additional columns
    #dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
    # print, to csv
    #print(dataset)
    #dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}.csv", index=False)

    return df


#df = getHistoricalData('BTC_USDT', start=1694264564-(60*5*1600), end=1694264564, candleIntv='MINUTE_5')
#print(df)
















































#
#
#
#START = 1467331200 # 01.07.2016 dd.mm.yyyy
#END = 1588284000 # 01.05.2020
#END = 1680367000 # 01.04.2023
#
#CANDLES_PERIOD = 21600
#
#
## gets historical chart data from Poloniex API (300s candles)
#def get_ChartData(coin, start, end):
#    raw = polo.returnChartData(f"USDT_{coin}", CANDLES_PERIOD, start, end)
#    df = pd.DataFrame(raw)
#    df.rename(columns={"date":"timestamp", "close":f"{coin}_close", "open":f"{coin}_open", "low":f"{coin}_low", "high":f"{coin}_high", "quoteVolume":f"{coin}_volume", "weightedAverage":f"{coin}_average"}, inplace=True)
#    #df.set_index("date", inplace=True)
#    # select columns to be used
#    df = df[["timestamp", f"{coin}_close", f"{coin}_low", f"{coin}_high", f"{coin}_volume", f"{coin}_average"]]
#    df["timestamp"] = df["timestamp"]/1000
#    #print("len api response:", len(df))
#    return df
#
#
#
#dataset = pd.DataFrame()
#
#
#
#
#
#
## collect data in 500 rows intervals (the API doesn't allow larger requests)
#intervalLength = 500*CANDLES_PERIOD
#intervalStart = START
#intervalEnd = START + intervalLength
#intervalsCounter = 1
#numIntervals = int((END - START) / intervalLength)
#
#
## for when API breaks access
### load DF
##intervalsCounter = 150
##dataset = pd.read_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv")
##intervalStart = START + (intervalLength * intervalsCounter)
##intervalEnd = intervalStart + intervalLength
##intervalsCounter += 1
#
#while(intervalEnd < END):
#    dataset = pd.concat([dataset, get_ChartData("BTC", intervalStart, intervalEnd)], ignore_index=True)
#    # shift interval
#    intervalStart = intervalEnd
#    intervalEnd += intervalLength
#    # counter
#    print("intervals: ", intervalsCounter, "/", numIntervals, " len dataset: ", len(dataset))
#    
#    if intervalsCounter % 50 == 0:
#        dataset = dataset.apply(pd.to_numeric)
#        # add additional columns
#        dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
#        # print, to csv
#        dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}_{intervalsCounter}.csv", index=False)
#        time.sleep(60)
#    intervalsCounter += 1
#
#intervalEnd = END
#dataset = pd.concat([dataset, get_ChartData("BTC", intervalStart, intervalEnd)], ignore_index=True)
#
#
#dataset = dataset.apply(pd.to_numeric)
## add additional columns
#dataset["BTC_HLPercent"] = (dataset["BTC_high"] - dataset["BTC_low"]) / dataset["BTC_high"]
## print, to csv
#print(dataset)
#dataset.to_csv(f"HistoricalData_2016_2023_{CANDLES_PERIOD}.csv", index=False)