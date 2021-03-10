from poloniex import Poloniex
from sklearn import preprocessing
import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

polo = Poloniex()

SEQ_LEN = 240
# load model
modelName = "r20t0-18"
model = tf.keras.models.load_model(""+modelName+".h5")


START = 1590969600 # 01.06.2020 dd.mm.yyyy
END = 1591747200 # 10.06.2020


def combine_dfs(list_dfs):
    df = pd.DataFrame()
    for list_df in list_dfs:
        if len(df) == 0:
            df = list_df
        else:
            df = df.join(list_df)
    return df
    

def get_ChartData(coin):
    while True:
        try:
            raw = polo.returnChartData(f"USDT_{coin}", 300, START, END)
        except:
            print("connection lost, trying again")
            time.sleep(60)
            pass
        else:
            # connected
            break
    df = pd.DataFrame(raw)
    df.rename(columns={"close": f"{coin}_close", "low": f"{coin}_low", "high": f"{coin}_high", "quoteVolume": f"{coin}_volume", "weightedAverage": f"{coin}_average"}, inplace=True)
    df = df[[f"{coin}_volume", f"{coin}_low", f"{coin}_high", f"{coin}_close", f"{coin}_average"]]
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



# DF
main_df = get_ChartData("BTC")
main_df = main_df.astype(float)
# additional columns
main_df["BTC_HLPercent"] = (main_df["BTC_high"] - main_df["BTC_low"]) / main_df["BTC_high"]
# right order (same columns and order as trained on)
main_df = main_df[["BTC_close","BTC_low","BTC_high","BTC_volume", "BTC_average", "BTC_HLPercent"]]

main_df = main_df.replace([0.0], 0.0001)
main_df.index = np.arange(0, len(main_df))


# for plotting
prices = main_df["BTC_close"].to_list()
prices = [float(price) for price in prices]
prices = [round(price, 2) for price in prices]
buyTimes = []
buyPrices =  []
sellTimes = []
sellPrices = []
holdTimes = []
holdPrices = []
confidences = []

# wallet simulation
usd = 50
eth = 50/prices[SEQ_LEN]
percentage = 0.10 #buy/sell percentage (of available balance)


# simulation
for i in tqdm(range(0, len(main_df) - SEQ_LEN)):

    # get current df
    current_df = main_df.head(SEQ_LEN + len(main_df.columns) + i).tail(SEQ_LEN + len(main_df.columns)).copy()
    current_df.index = np.arange(0, len(current_df))
    current_price = current_df["BTC_close"][SEQ_LEN + len(main_df.columns) - 1]
    # preprocess df
    current_df = preprocessDf(current_df)
    # build sequence
    current_sequence = buildSequence(current_df)
    # predict
    prediction_confs = model.predict(current_sequence)[0]
    # select max conf
    prediction = [np.argmax(prediction_confs), np.max(prediction_confs)]
    confidences.append(prediction[1])
    # execute decision
    if prediction[0] == 1:
        # buy
        buyTimes.append(i + len(main_df.columns) - 1)
        buyPrices.append(current_price)
        # wallet simulation
        buyDollar = usd*percentage
        usd = usd - buyDollar
        eth = eth + (buyDollar/current_price)*1 #0.9991 # fees

    elif prediction[0] == 0:
        #sell
        sellTimes.append(i + len(main_df.columns) - 1)
        sellPrices.append(current_price)
        # wallet simulation
        sellEth = eth*percentage
        eth = eth - sellEth
        usd = usd + (current_price*sellEth)*1 #0.9991 # fees
    elif prediction[0] == 2:
        # hold
        holdTimes.append(i + len(main_df.columns) - 1)
        holdPrices.append(current_price)




# stats
averageBuy = np.mean(buyPrices)
averageSell = np.mean(sellPrices)
print("buys: ", len(buyPrices), ", average: ", averageBuy)
print("sells: ", len(sellPrices), ", average: ", averageSell)
print("result: ", ((eth*prices[-1] + usd)/100))
print("market:  ", (prices[-1]/prices[SEQ_LEN]))
print("delta: ", (((eth*prices[-1] + usd)/100) - (prices[-1]/prices[SEQ_LEN])))


outputDF = pd.DataFrame()
outputDF["times"] = buyTimes+sellTimes+holdTimes
outputDF["times"] = outputDF["times"]-len(main_df.columns)+1
outputDF["prices"] = buyPrices+sellPrices+holdPrices
outputDF["sellBuyHold"] = np.concatenate([np.ones_like(buyTimes)*1,np.ones_like(sellTimes)*0,np.ones_like(holdTimes)*2])
outputDF.sort_values(by=["times"], inplace=True)
outputDF["confidence"] = confidences
outputDF.set_index("times", inplace=True)
print(outputDF)
# to csv
outputDF.to_csv("modelOutput_"+modelName+".csv")

#plot
prices = prices[SEQ_LEN:]
plt.plot(prices)
plt.plot(buyTimes, buyPrices, 'go')
plt.plot(sellTimes, sellPrices, 'ro')
plt.show()
