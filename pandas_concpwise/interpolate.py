#interplote vs fillna: interpolta is stronger compared to fillna

import pandas as pd 
import numpy as np 
df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

df.interpolate(inplace = True)
print(df)

df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

df = df.interpolate(method = 'linear', limit_direction = 'forward', axis = 0)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

df = df.interpolate(method = 'linear', limit_direction = 'backward',axis = 0)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

df = df.interpolate(method = 'polynomial', order = 2,  limit_direction = 'backward',axis = 0)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

df = df.interpolate(method = 'polynomial', order = 3, limit_direction = 'forward',axis = 0)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

#fill NULLs with ZEROs 
df = df.fillna(0)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

#fill NULLs with ONEs
df = df.fillna(1)
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

#fill NULLs with eixsting non-null preceding value
df = df.fillna(method = "ffill")
print(df)


df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])
print(df)

#fill NULLs with eixsting non-null succeding value
df = df.fillna(method = "bfill")
print(df)


df = pd.DataFrame([[np.nan, 2, np.nan, np.nan],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns = list('ABCD'))
print(df)

values = {'A': 100, 'B': 200, 'C': 300, 'D': 400}
df = df.fillna(value = values)
print(df)


df = pd.DataFrame([[np.nan, 2, np.nan, np.nan],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))
print(df)


values = {'A': 100, 'B': 200, 'C': 300, 'D': 400}
df = df.fillna(value=values, limit=2)
print(df)