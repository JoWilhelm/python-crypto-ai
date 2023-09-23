#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import preprocessing


SEQ_LEN = 180 #240   # how many past candles to use to predict
CANDLES_SHIFT = 10000#2 #5 # how many candles to shift between sequences
NAME = "m5_ov40th04p_shift2_seq180"
VALIDATION_PCT = 0.2



def preprocess2(dfs):
    # after sequencing
    #
    # pct.change transform price columns ('low', 'high', 'open', 'close')
    # scale every colum (center mean and unit variance)

    for df in dfs:
        for col in df.columns:
            if col != "target":
                if col != 'quantity_baseUnits' and col != 'hl_percent':
                    df[col] = df[col].pct_change()
                    df.dropna(inplace=True)
                df[col] = preprocessing.scale(df[col].values)
                df.index = np.arange(0, len(df))
    return dfs


def splitDf(df):
    res = []
    print("")
    print("splitDf")
    while len(df) >= SEQ_LEN + len(df.columns) -1:
        first = df.head(SEQ_LEN + len(df.columns) -1).copy()
        first.index = np.arange(0, len(first))
        res.append(first)
        df = df.tail(len(df) - CANDLES_SHIFT)
        df.index = np.arange(0, len(df))

    print("-done")
    print("")
    return res



#def buildSequences(dfs):
#    sequences = []
#    for df in dfs:
#        if(len(df) == SEQ_LEN):
#            label = df.at[SEQ_LEN-1, 'target']
#            df = df.iloc[:, :-1]
#            dfArray = df.values.tolist()
#            sequences.append([np.array(dfArray), label])
#    
#    return sequences





# load data
df = pd.read_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv")
df = df[['close', 'hl_percent', 'quantity_baseUnits']]
train_size = int((1-VALIDATION_PCT) * len(df))
#df = df.tail(5000)
#print(df)


# split into sequences
splittedDfs = splitDf(df)
#print("num sequences:", len(splittedDfs))
#print(splittedDfs[0])

# preprocess
splittedDfs = preprocess2(splittedDfs)
#print(splittedDfs[0])



# train val split
dfsTraining = splittedDfs[:(int(len(splittedDfs) * (1-VALIDATION_PCT)))].copy()
dfsValidation = splittedDfs[(int(len(splittedDfs) * (1-VALIDATION_PCT))):].copy()
#print(len(dfsTraining))
#print(len(dfsValidation))
#print(dfsTraining[0])




#sequencesTraining = buildSequences(dfsTrainingPreprocessed)
#sequencesValidation = buildSequences(dfsValidationPreprocessed)








## ANIMATED SEQ PLOT

## Initialize the plot
#fig, ax = plt.subplots()
#
## Initialize a line plot. Note that the data for the line will be updated later.
#line, = ax.plot([], [], lw=2)
#
## Function to initialize the plot (required by FuncAnimation)
#def init():
#    line.set_data([], [])
#    return line,
#
## Function to update each frame of the plot
#def update(frame):
#    xdata = range(frame*CANDLES_SHIFT, len(splittedDfs[frame]['close']) + frame*CANDLES_SHIFT)
#    ydata = splittedDfs[frame]['close']
#    ax.set_xlim(frame*CANDLES_SHIFT, len(xdata) - 1 + frame*CANDLES_SHIFT)
#    ax.set_ylim(-5, 5)  # Optionally set y-limits to adjust automatically
#    line.set_data(xdata, ydata)
#    ax.relim()  # Recompute the axis limits
#    ax.autoscale_view()  # Adjust the axis limits based on the data
#    plt.draw()  # Force a draw event
#    return line,
#
## Create an animation
#ani = FuncAnimation(
#    fig,
#    update,
#    frames=range(len(splittedDfs)),  # Number of frames
#    init_func=init,
#    blit=False,
#    interval=200  # Show each frame for 0.5 seconds (500 milliseconds)
#)
#
#plt.show()
#




## PLOT INDIVIDUAL SEQ

seq = 705600
seqIndx = int(seq / CANDLES_SHIFT)

## FANCY PLOTS

# Create subplots in a vertical layout
fig, axes = plt.subplots(nrows=len(splittedDfs[seqIndx].columns), ncols=1, figsize=(5, 15))

# If there's only one subplot, axes will not be an array; convert it to an array for consistency
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Create x-values
x = np.arange(seq, len(splittedDfs[seqIndx]) + seq)

for i, column in enumerate(splittedDfs[seqIndx].columns):

    if seq > train_size:
        c = 'lightblue'
        l = 'Validation'
    else:
        c = 'tab:blue'
        l = 'Train'
    # Plot the data
    axes[i].plot(x, splittedDfs[seqIndx][column], color=c, label=l)
    
    # Add title and legend to each subplot
    axes[i].set_title(column)

# Create one legend for the entire figure
handles, labels = axes[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')  # You can adjust the 'loc' parameter as needed

# Adjust layout to add vertical white space between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, top=0.92)  # Adjust the vertical spaces between plots and the top margin

plt.show()

