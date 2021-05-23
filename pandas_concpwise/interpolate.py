# df_fcst_mau['incr_mau'].fillna(0, inplace=True)

import pandas as pd 
import numpy as np 
df = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan])

print(df)

# df.interpolate(inplace = True)
print(df)
# df = df.interpolate(method='linear', limit_direction='forward', axis=0)
# print(df)
df = df.interpolate(method='linear', limit_direction='backward',axis=0)
print(df)
# df = df.interpolate(method='polynomial', order =2,  limit_direction='backward',axis=0)
# print(df)
df = df.interpolate(method='polynomial', order =2, limit_direction='forward',axis=0)
print(df)
