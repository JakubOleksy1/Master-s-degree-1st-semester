import pandas as pd
import quandl

# Set Quandl API key if you have one
quandl.ApiConfig.api_key = "764YURLHwMeYZpuLPdr_"

# Fetch Bitcoin blockchain data from Quandl
data = quandl.get("WIKI/GOOGL", paginate=True)  # Adjust the dataset code as needed

data = data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100.0
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100.0

data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
print(data.head())
