import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



SEQ_LEN = 180 #240   # how many past candles to use to predict
CANDLES_SHIFT = 50#2 #5 # how many candles to shift between sequences
NAME = "m5_ov40th04p_shift2_seq180"
VALIDATION_PCT = 0.2


# Function to preprocess data
def preprocess1_train(df):
    # before sequencing
    #
    # pct.change transform price columns ('low', 'high', 'open', 'close')
    # scale every colum (center mean and unit variance)

    scaler_dict = {}
    for col in df.columns:
        if col != 'target':
            if col != 'quantity_baseUnits' and col != 'hl_percent':
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            scaler_dict[col] = scaler
    df.index = np.arange(0, len(df))
    return df, scaler_dict

# Function to apply saved preprocessing to new data
def apply_preprocess1_val(df, scaler_dict):
    # before sequencing
    #
    # pct.change transform price columns ('low', 'high', 'open', 'close')
    # scale every colum (center mean and unit variance)
    
    for col in df.columns:
        if col != 'target':
            if col != 'quantity_baseUnits' and col != 'hl_percent':
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)
            scaler = scaler_dict[col]
            df[col] = scaler.transform(df[col].values.reshape(-1, 1))
    df.index = np.arange(0, len(df))
    return df


def splitDf_new(df):
    
    res = []
    print("")
    print("splitDf")
    while len(df) >= SEQ_LEN:
        first = df.head(SEQ_LEN).copy()
        first.index = np.arange(0, len(first))
        res.append(first)
        df = df.tail(len(df) - CANDLES_SHIFT)
        df.index = np.arange(0, len(df))

    print("-done")
    print("")
    return res





# load data
df = pd.read_csv("historicalData/labeled/HistoricalDataLabeled_BTC_USDT_01072016_01072023_MINUTE_5_ov40_th04p.csv")
df = df[['close', 'hl_percent', 'quantity_baseUnits']]
#df = df.tail(5000)
print(df)

# Split data into train and validation sets
train_size = int((1-VALIDATION_PCT) * len(df))
train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:].copy()

# Preprocess the training data and save the scaling parameters
train_df, scaler_dict = preprocess1_train(train_df)
# Apply saved preprocessing to validation data
val_df = apply_preprocess1_val(val_df, scaler_dict)




train_dfs = splitDf_new(train_df)
val_dfs = splitDf_new(val_df)

splittedDfs = train_dfs + val_dfs












#
### FANCY PLOTS
#
## Create subplots in a vertical layout
#fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(5, 15))
#
## If there's only one subplot, axes will not be an array; convert it to an array for consistency
#if not isinstance(axes, np.ndarray):
#    axes = np.array([axes])
#
## Create x-values for the train and validation data
#train_x = np.arange(len(train_df))
#val_x = np.arange(len(train_df), len(train_df) + len(val_df))
#
#for i, column in enumerate(df.columns):
#    # Plot the train data
#    axes[i].plot(train_x, train_df[column], label='Train')
#    
#    # Plot the validation data right next to the train data
#    axes[i].plot(val_x, val_df[column], label='Validation', color='lightblue')
#    
#    # Add title and legend to each subplot
#    axes[i].set_title(column)
#
## Create one legend for the entire figure
#handles, labels = axes[i].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right')  # You can adjust the 'loc' parameter as needed
#
## Adjust layout to add vertical white space between subplots
#plt.tight_layout()
#plt.subplots_adjust(hspace=0.3, top=0.92)  # Adjust the vertical spaces between plots and the top margin
#
#plt.show()
#






## PLOT INDIVIDUAL SEQ

seq = 107700
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
