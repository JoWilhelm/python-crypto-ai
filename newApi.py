from polosdk import RestClient
import pandas as pd

client = RestClient()




# Returns OHLC for a symbol at given timeframe (interval)
response = client.markets().get_candles('BTC_USDT', 'MINUTE_5', start=1694264564-(60*5*500), end=1694264564, limit=500)
df = pd.DataFrame(response, columns=['low', 'high', 'open', 'close', 'amount_quoteUnits', 'quantity_baseUnits', 'buyTakerAmount_quoteUnits', 'buyTakerQuantity_baseUnits', 'tradeCount', 'ts_recordPushed', 'weightedAverage', 'interval', 'startTime', 'closeTime'])






print(df)