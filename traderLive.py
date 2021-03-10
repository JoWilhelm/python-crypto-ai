import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from poloniex import Poloniex
import sys
import math
import os
import os.path

#api_key = 'ABC'
#api_secret = '123'
#polo = Poloniex(api_key, api_secret)
polo = Poloniex()


# gets dataframe from poloniex api
def getChartData(coin, length, candles):
    end = int(time.time())
    start = end - length
    while True:
        try:
            raw = polo.returnChartData(f"USDT_{coin}", candles*60, start, end)
        except:
            print("connection lost, trying again")
            time.sleep(60)
            pass
        else:
            # connected
            break
    df = pd.DataFrame(raw)
    df.rename(columns={"close": f"{coin}_close", "low": f"{coin}_low", "high": f"{coin}_high", "quoteVolume": f"{coin}_volume", "weightedAverage": f"{coin}_average"}, inplace=True)
    df = df[[f"{coin}_close", f"{coin}_low", f"{coin}_high", f"{coin}_volume", f"{coin}_average"]]
    return df

# puts dataframes side by side
def combine_dfs(list_dfs):
    df = pd.DataFrame()
    for list_df in list_dfs:
        if len(df) == 0:
            df = list_df
        else:
            df = df.join(list_df)
    return df

def preprocessDf(df):
    for col in df.columns:
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values)
        df.index = np.arange(0, len(df))
    return df

def buildSequence(df):
    sequence = []
    dfArray = df.values.tolist()
    sequence.append(np.array(dfArray))
    return np.array(sequence)

def current_price(coin):
    try:
        return float(polo.returnTicker()[f"USDT_{coin}"]["last"])
    except:
        print("connection lost, restarting - current price")
        time.sleep(30)
        python = sys.executable
        os.execl(python, python, *sys.argv)


class Strategy():
    SEQ_LEN = 240
    # load model(-s)
    model1 = tf.keras.models.load_model("r20t0-18.h5")
    
    tradingPercentage = 0.10 #buy/sell percentage (of available balance)
    pastConfs = deque(maxlen=300)
    
    def predict(self, sequence):
        prediction_confs_model1 = self.model1.predict(current_sequence)[0]
        prediction_model1 = [np.argmax(prediction_confs_model1), np.max(prediction_confs_model1)]
        # only buy/sell if confidence higher than average
        self.pastConfs.append(prediction_model1[1])
        pastConfsAverage = np.mean(self.pastConfs) 
        if prediction_model1[1] >= pastConfsAverage:
            return prediction_model1[0]
        else: 
            return 2
    
    # places buy order
    def buy(self):
        #buy
        currentPrice = currentPrice("BTC")
        usd = float(polo.returnBalances()['USDT'])
        buyAmount = usd*tradingStrat.tradingPercentage/currentPrice
        #polo.buy('USDT_BTC', currentPrice, buyAmount)
        print("b @", round(currentPrice, 2))
    
    # places sell order
    def sell(self):
        #sell
        currentPrice = currentPrice("BTC")
        btc = float(polo.returnBalances()['ETH'])
        sellAmount = btc*tradingStrat.tradingPercentage
        #polo.sell('USDT_BTC', current_price("BTC"), sellAmount)
        print("s @", round(currentPrice, 2))
    




tradingStrat = Strategy()

while True:
    
    # time synchronisation to next candle close
    m = int(math.floor(time.time()/300))
    while int(math.floor(time.time()/300)) == m:
        time.sleep(1)
    
    # get current dataframe
    df = getChartData("BTC", tradingStrat.SEQ_LEN*5*60*1.5, 5)
    df = df.astype(float)
    df["BTC_HLPercent"] = (df["BTC_high"] - df["BTC_low"]) / df["BTC_high"]
    df = df[["BTC_close", "BTC_low", "BTC_high", "BTC_volume", "BTC_average", "BTC_HLPercent"]]
    df = df.replace([0.0], 0.0001)
    current_df = df.tail(tradingStrat.SEQ_LEN + len(df.columns)).copy()
    current_df.index = np.arange(0, len(current_df))
    # replace last price in DF with most recent price
    current_df["BTC_close"][tradingStrat.SEQ_LEN + len(df.columns) - 1] = current_price("BTC")

    # preprocess df
    current_df = preprocessDf(current_df)
    # build sequence
    current_sequence = buildSequence(current_df)

    # check for correct shape
    if current_sequence.shape != (1, tradingStrat.SEQ_LEN, len(df.columns)):
        # wrong input shape - restart
        print("wrong input shape - restarting")
        time.sleep(10)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    # predict
    prediction = tradingStrat.predict
    
    # execute decision
    if prediction== 1:
        # buy
        tradingStrat.buy()
    elif prediction== 0:
        #sell
        tradingStrat.sell()
    else:
        # hold
        print("-")
    
    
    # end of loop, sleep until next candle close
    time.sleep(280)
     




