from polosdk import RestClient
import pandas as pd
from datetime import datetime


client = RestClient()






# Returns OHLC for a symbol at given timeframe (interval)
response = client.markets().get_candles('BTC_USDT', 'MINUTE_5', start=1694264564-(60*5*500), end=1694264564, limit=500)


df = pd.DataFrame(response, columns=['low', 'high', 'open', 'close', 'amount_quoteUnits', 'quantity_baseUnits', 'buyTakerAmount_quoteUnits', 'buyTakerQuantity_baseUnits', 'tradeCount', 'ts_recordPushed', 'weightedAverage', 'interval', 'ts_startTime', 'ts_closeTime'])
df.drop('interval', axis=1, inplace=True)
df.apply(pd.to_numeric)

df['dt_close'] = [datetime.fromtimestamp(ts/1000) for ts in df['ts_closeTime']]


#print(df['ts_startTime'])
print((df['ts_recordPushed'] - df['ts_startTime'])/1000)






#fig = go.Figure(data=[go.Candlestick(x=df['dateTime'],
#                       open=df['open'], high=df['high'],
#                       low=df['close'], close=df['close'])])
#
#fig.show()