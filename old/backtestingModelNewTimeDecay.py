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

SEQ_LEN = 60
# load model
modelName = "decay_seq60_ov40th004-36"
model = tf.keras.models.load_model("models/"+modelName+".h5")


START = 1680369000 # 01.04.2023 dd.mm.yyyy
START = 1682963000 # 01.05.2023
#START = 1685223000 # 27.05.2023
#START = 1685398000 # 30.05.2023
END = 1685640000 # 01.06.2023




means_300 = {"BTC_close": 0.000000690947, 
             "BTC_low": 0.0000006854884, 
             "BTC_high": 0.0000006769006, 
             "BTC_volume": 9.29817, 
             "BTC_HLPercent": 0.002693165}
stds_300 = {"BTC_close": 0.0003184924, 
             "BTC_low": 0.0002982297, 
             "BTC_high": 0.0002733320261, 
             "BTC_volume": 2.81534, 
             "BTC_HLPercent": 0.00318942}

means_900 = {"BTC_close": 0.0000020544577, 
             "BTC_low": 0.000002046661877, 
             "BTC_high": 0.00000201864122,  
             "BTC_volume": 10.884899, 
             "BTC_HLPercent": 0.005028761}
stds_900 = {"BTC_close": 0.000519522, 
             "BTC_low": 0.00050422044, 
             "BTC_high": 0.0004491347, 
             "BTC_volume": 2.2087313, 
             "BTC_HLPercent": 0.005339183}

means_3600 = {"BTC_close": 0.00000814210187, 
             "BTC_low": 0.0000081886629, 
             "BTC_high": 0.0000079834025, 
             "BTC_volume": 12.5666656, 
             "BTC_HLPercent": 0.01051739}
stds_3600 = {"BTC_close": 0.00098108448, 
             "BTC_low": 0.0010199776, 
             "BTC_high": 0.000852822218, 
             "BTC_volume": 1.83560254, 
             "BTC_HLPercent": 0.0102345275}

means_21600 = {"BTC_close": 0.0000483701454, 
             "BTC_low": 0.0000490174551, 
             "BTC_high": 0.00004777989, 
             "BTC_volume": 14.559, 
             "BTC_HLPercent": 0.02660019}
stds_21600 = {"BTC_close": 0.0022161646, 
             "BTC_low": 0.002430053655, 
             "BTC_high": 0.00197076029, 
             "BTC_volume": 1.6195640409, 
             "BTC_HLPercent": 0.022443346231}




def get_ChartData(coin, candlesPeriod):
    if (END - START)/candlesPeriod <= 500:
        return get_ChartData_interval(coin, START, END, candlesPeriod)
    

 

    # collect data in 500 rows intervals (the API doesn't allow larger requests)
    intervalLength = 500*candlesPeriod
    intervalStart = START
    intervalEnd = START + intervalLength
    intervalsCounter = 1
    numIntervals = int((END - START) / intervalLength)


    dataset = pd.DataFrame()
    while(intervalEnd < END):
        dataset = pd.concat([dataset, get_ChartData_interval(coin, intervalStart, intervalEnd, candlesPeriod)], ignore_index=True)
        # shift interval
        intervalStart = intervalEnd
        intervalEnd += intervalLength
        # counter
        print("intervals: ", intervalsCounter, "/", numIntervals, " len dataset: ", len(dataset))
    
        if intervalsCounter % 50 == 0:
            time.sleep(60)
        intervalsCounter += 1

    intervalEnd = END
    dataset = pd.concat([dataset, get_ChartData_interval(coin, intervalStart, intervalEnd, candlesPeriod)], ignore_index=True)
    
    return dataset

    

def get_ChartData_interval(coin, intervalStart, intervalEnd, candlesPeriod):
    while True:
        try:
            raw = polo.returnChartData(f"USDT_{coin}", candlesPeriod, intervalStart, intervalEnd)
        except:
            print("connection lost, trying again")
            time.sleep(60)
            pass
        else:
            # connected
            break
    df = pd.DataFrame(raw)
    df.rename(columns={"date":"timestamp", "close": f"{coin}_close", "low": f"{coin}_low", "high": f"{coin}_high", "quoteVolume": f"{coin}_volume", "weightedAverage": f"{coin}_average"}, inplace=True)
    df = df[["timestamp", f"{coin}_close", f"{coin}_low", f"{coin}_high", f"{coin}_volume", f"{coin}_average"]]
    df["timestamp"] = df["timestamp"]/1000
    df["timescale"] = int(candlesPeriod/60)
    df["BTC_HLPercent"] = (df["BTC_high"] - df["BTC_low"]) / df["BTC_high"]
    return df



def preprocess(df, means, stds):

    for col in ["BTC_close", "BTC_low", "BTC_high"]:
        df[col] = np.log(df[col])
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        mean = means[col]
        std = stds[col]
        df[col] = (df[col] - mean) / std
    
    df["BTC_volume"] = df["BTC_volume"].replace(0, 1)
    df["BTC_volume"] = np.log(df["BTC_volume"])
    #df["BTC_volume"] = df["BTC_volume"].pct_change()   # taking the pct change somehow makes it worse
    #df.dropna(inplace=True)                            # taking the pct change somehow makes it worse
    mean = means["BTC_volume"]
    std = stds["BTC_volume"]
    df["BTC_volume"] = (df["BTC_volume"] - mean) / std

    mean = means["BTC_HLPercent"]
    std = stds["BTC_HLPercent"]
    df["BTC_HLPercent"] = (df["BTC_HLPercent"] - mean) / std

    return df


sequenceEndTimestamps = []
# composes dfs of individual sequences from different timescales (in reverse)
# dfs should be ordered fine to coarse with the finest one having the labels
# seqLen should be divisible by the number of time-resolutions len(dfs)
def composeTimeDecaySequences(dfs, timescales):
    numTimeScales = len(dfs)
    seqPartLen = int(SEQ_LEN/numTimeScales)

    sequences = []
    finestEndIndex = len(dfs[0])-1
    for _ in tqdm(range(int((len(dfs[0]) -int(seqPartLen*(sum(timescales))/300))))): # calculating how many sequences there will be
        
        lastTimestamp = dfs[0]["timestamp"].iloc[finestEndIndex]+1
        seqDf = pd.DataFrame()
        for i in range(numTimeScales):

            # get prev index. Last with timestamp < lastTimestamp
            endIndex = dfs[i][dfs[i]["timestamp"] < lastTimestamp].index[-1]
            seqDf = pd.concat([dfs[i][endIndex-seqPartLen:endIndex], seqDf])
            lastTimestamp = seqDf["timestamp"].iloc[0]
        

        endTimestamp = seqDf["timestamp"].iloc[-1]
        seqDf = seqDf[["BTC_close", "BTC_volume", "BTC_HLPercent", "timescale"]]
        seq = seqDf.values.tolist()
        if(len(seq) == SEQ_LEN):
            sequences.append(seq)
            sequenceEndTimestamps.append(endTimestamp)

        finestEndIndex -= 1

    return sequences[::-1]
    






print("fetching market data...")
# fetch DFs
df_300 = get_ChartData("BTC", 300)
df_900 = get_ChartData("BTC", 900)
df_3600 = get_ChartData("BTC", 3600)
df_21600 = get_ChartData("BTC", 21600)

# right order (same columns and order as trained on)
df_300 = df_300[["timestamp", "BTC_close","BTC_low","BTC_high","BTC_volume", "BTC_HLPercent", "timescale"]]
df_900 = df_900[["timestamp", "BTC_close","BTC_low","BTC_high","BTC_volume", "BTC_HLPercent", "timescale"]]
df_3600 = df_3600[["timestamp", "BTC_close","BTC_low","BTC_high","BTC_volume", "BTC_HLPercent", "timescale"]]
df_21600 = df_21600[["timestamp", "BTC_close","BTC_low","BTC_high","BTC_volume", "BTC_HLPercent", "timescale"]]

# manually align dfs
for col in df_300.columns:
    if not col == "timestamp":
        df_300[col] = df_300[col].shift(-71)
for col in df_900.columns:
    if not col == "timestamp":
        df_900[col] = df_900[col].shift(-23)
for col in df_3600.columns:
    if not col == "timestamp":
        df_3600[col] = df_3600[col].shift(-5)

df_300 = df_300.dropna()
df_900 = df_900.dropna()
df_3600 = df_3600.dropna()
df_21600 = df_21600.dropna()

#print(df_300)
#print(df_900)
#print(df_3600)
#print(df_21600)
#
#plt.plot(df_300["timestamp"], df_300["BTC_close"])
#plt.plot(df_900["timestamp"], df_900["BTC_close"])
#plt.plot(df_3600["timestamp"], df_3600["BTC_close"])
#plt.plot(df_21600["timestamp"], df_21600["BTC_close"])
#
#plt.show()


og_prices_300 = df_300[["timestamp", "BTC_close"]].copy()

# preprocessing
df_300_pp = preprocess(df_300, means_300, stds_300)
df_900_pp = preprocess(df_900, means_900, stds_900)
df_3600_pp = preprocess(df_3600, means_3600, stds_3600)
df_21600_pp = preprocess(df_21600, means_21600, stds_21600)
#print(df_300_pp)

# build sequences
print("building time decay sequences...")
sequences = composeTimeDecaySequences([df_300_pp, df_900_pp, df_3600_pp, df_21600_pp], [300, 900, 3600, 21600])
sequenceEndTimestamps = sequenceEndTimestamps[::-1]




# for plotting
prices = og_prices_300[og_prices_300['timestamp'].isin(sequenceEndTimestamps)]['BTC_close']
prices = prices.to_list()
buyTimes = []
buyPrices =  []
sellTimes = []
sellPrices = []
holdTimes = []
holdPrices = []
confidences = []






# simulation
for i in tqdm(range(len(sequences))):

    current_sequence = sequences[i]
    current_price = prices[i]

    # predict
    prediction_confs = model.predict([current_sequence], verbose=0)[0]
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