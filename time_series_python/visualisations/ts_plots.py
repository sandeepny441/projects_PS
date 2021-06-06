import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("stock_data.csv", parse_dates=True, index_col = "Date")
df.head()

df['Volume'].plot()
df.plot(subplots=True, figsize=(10,12))
print(df.head())



df_month = df.resample("M").mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.bar(df_month['2016':].index, df_month.loc['2016':, "Volume"], width=25, align='center')

plt.show()