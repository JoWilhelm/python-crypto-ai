from poloniex import Poloniex
from sklearn import preprocessing
import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

polo = Poloniex()

SEQ_LEN = 500
# load model
modelName = "ov40th005-07"
model = tf.keras.models.load_model("models/"+modelName+".h5")


START = 1680369000 # 01.04.2023 dd.mm.yyyy
#START = 1685223000 # 27.05.2023
#START = 1685398000 # 30.05.2023
END = 1685640000 # 01.06.2023


def combine_dfs(list_dfs):
    df = pd.DataFrame()
    for list_df in list_dfs:
        if len(df) == 0:
            df = list_df
        else:
            df = df.join(list_df)
    return df




def get_ChartData(coin):
    if END - START <= 129600:
        return get_ChartData_interval(coin, START, END)
    

    # collect data in 1.5d intervals (the API doesn't allow larger requests)
    intervalStart = START
    intervalEnd = START + 129600 # +1.5d
    intervalsCounter = 1

    dataset = pd.DataFrame()
    while(intervalEnd < END):
        dataset = pd.concat([dataset, get_ChartData_interval(coin, intervalStart, intervalEnd)], ignore_index=True)
        # shift interval 1.5d
        intervalStart = intervalEnd
        intervalEnd += 129600 # +1.5d
        # counter
        print("intervals: ", intervalsCounter, "/", int((END-START)/129600), " len dataset: ", len(dataset))

        if intervalsCounter % 50 == 0:
            time.sleep(60)
        intervalsCounter += 1

    intervalEnd = END
    dataset = pd.concat([dataset, get_ChartData_interval(coin, intervalStart, intervalEnd)], ignore_index=True)
    return dataset

    

def get_ChartData_interval(coin, intervalStart, intervalEnd):
    while True:
        try:
            raw = polo.returnChartData(f"USDT_{coin}", 300, intervalStart, intervalEnd)
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


def preprocessDf_new(df):
    
    for col in ["BTC_close", "BTC_low", "BTC_high", "BTC_average"]:
        #print(col)
        df[col] = np.log(df[col])
        df[col] = df[col].pct_change()
        #df.dropna(inplace=True)
        #mean = np.mean(df[col])
        mean = 0.00000068
        #std = np.std(df[col])
        std = 0.00028
        #print("mean:", mean, ", std:", std)
        df[col] = (df[col] - mean) / std
    df.dropna(inplace=True)
    
    df["BTC_volume"] = df["BTC_volume"].replace(0, 1)
    df["BTC_volume"] = np.log(df["BTC_volume"])
    #df["BTC_volume"] = df["BTC_volume"].pct_change()   # taking the pct change somehow makes it worse
    #df.dropna(inplace=True)                            # taking the pct change somehow makes it worse
    #mean = np.mean(df["BTC_volume"])
    mean = 9.3
    #std = np.std(df["BTC_volume"])
    std = 2.82
    #print("mean:", mean, ", std:", std)
    df["BTC_volume"] = (df["BTC_volume"] - mean) / std


    #mean = np.mean(df["BTC_HLPercent"])
    mean = 0.0027
    #std = np.std(df["BTC_HLPercent"])
    std = 0.0032
    #print("mean:", mean, ", std:", std)
    df["BTC_HLPercent"] = (df["BTC_HLPercent"] - mean) / std


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
main_df.index = np.arange(0, len(main_df))


# for plotting
prices = main_df["BTC_close"].to_list()
prices = [float(price) for price in prices]
prices = prices[SEQ_LEN:]
buyTimes = []
buyPrices =  []
sellTimes = []
sellPrices = []
holdTimes = []
holdPrices = []
confidences = []

main_df = preprocessDf_new(main_df)

# simulation
for i in tqdm(range(0, len(main_df) - SEQ_LEN)):

    # get current df
    current_df = main_df.head(SEQ_LEN + i).tail(SEQ_LEN).copy()
    current_df.index = np.arange(0, len(current_df))
    #current_price = current_df["BTC_close"][SEQ_LEN -1]
    current_price = prices[i]
    # build sequence
    current_sequence = buildSequence(current_df)
    # predict
    prediction_confs = model.predict(current_sequence, verbose=0)[0]
    # select max conf
    prediction = [np.argmax(prediction_confs), np.max(prediction_confs)]
    confidences.append(prediction[1])
    # execute decision
    if prediction[0] == 1:
        # buy
        buyTimes.append(i)
        buyPrices.append(current_price)

    elif prediction[0] == 0:
        #sell
        sellTimes.append(i)
        sellPrices.append(current_price)
    elif prediction[0] == 2:
        # hold
        holdTimes.append(i)
        holdPrices.append(current_price)




# stats
averageBuy = np.mean(buyPrices)
averageSell = np.mean(sellPrices)
print("buys: ", len(buyPrices), ", average: ", averageBuy)
print("sells: ", len(sellPrices), ", average: ", averageSell)


outputDF = pd.DataFrame()
outputDF["times"] = buyTimes+sellTimes+holdTimes
#outputDF["times"] = outputDF["times"]-len(main_df.columns)+1
outputDF["prices"] = buyPrices+sellPrices+holdPrices
outputDF["sellBuyHold"] = np.concatenate([np.ones_like(buyTimes)*1,np.ones_like(sellTimes)*0,np.ones_like(holdTimes)*2])
outputDF.sort_values(by=["times"], inplace=True)
outputDF["confidence"] = confidences
outputDF.set_index("times", inplace=True)
print(outputDF)
# to csv
outputDF.to_csv("modelOutput_"+modelName+".csv")

#plot
#plt.plot(prices)
#plt.plot(buyTimes, buyPrices, 'go', markersize=confidences)
#plt.plot(sellTimes, sellPrices, 'ro', markersize=confidences)
#plt.show()

markerSizes = confidences
sns.lineplot(x=np.arange(len(prices)), y=prices, zorder=1)
sns.scatterplot(x=buyTimes, y=buyPrices, size=[markerSizes[i] for i in buyTimes], color="green", zorder=2, legend=False)
sns.scatterplot(x=sellTimes, y=sellPrices, size=[markerSizes[i] for i in sellTimes], color="red", zorder=2, legend=False)

plt.show()