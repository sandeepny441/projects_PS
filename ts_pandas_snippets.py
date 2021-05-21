import pandas as pd
import numpy as np

pd.set_option("display.precision", 3)
pd.set_option("display.max_rows", 25)

# creation:
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], index=[1000, 2000, 3000]})
df = pd.DataFrame([[1, 2, 3], [10, 20, 30], index=[40, 41], columns=['a', 'b', 'c']])

# loading_data
df = pd.read_csv('titanic.csv')
df = pd.read_csv('titanic.csv', skiprows=1)
df = pd.read_excel('titanic.excel')
df = pd.read_html('titanic.html')

# most used:
df.head()
df.tail()
df.sample(5)
df.sample(frac=0.3)
df.nlargest(10, 'A')
df.smallest(10, 'B')

# info:
df.info()
df.shape
df.describe()
df.dtypes
df['A'].value_counts()
df['A'].count()
df['A'].unique()
df.drop_duplicates()
df.values()
df.filter(items=['A'])
df.filter(regex='s$')  # column_name_match
df.filter(like='san')  # row_name_match

# columns:
df.columns
df.rename(columns={'A': 'B'})
df.drop(['A'], inplace=True)
df.columns = ['A', 'B', 'C']

# rows:
for i, row in df.iterrows():
    print(i, row)


# null_values:
df.isna()
df.isnull()
df.dropna()
df.fillna(10)
df.interpolate()
df.interpolate(method='linear', limit_direction='forward')
df.interpolate(method='linear', limit_direction='backward')
df.interpolate(method='linear', limit_direction='both')


# selection:
df.loc[1:100, 2:4]
df.loc[1:100, 'A']
df.loc[1:100, ['A', 'B']]

#selectiuon with information at location
df.iloc[1:100, 2:5]
df.loc[df['A'] > 100]
df['A'].loc[lambda x: x > 100]
df1 = df[['A', 'B']]

# concat  || Increasing Height 
pd.concat([df1, df2], axis=0)
pd.concat([df1, df2], axis=1)

# merge | joins in SQL:|| Increasing with  
pd.merge(df1, df2)
pd.merge(df1, df2, how='inner', on='a')
pd.merge(df1, df2, how='left', on='a')
pd.merge(df1, df2, how='right', on='a')
pd.merge(df1, df2, how='outer', on='a')
#how to include condition like we include where condition in SQL?

# sort:
df['A'].sort_values()
df['A'].sort_values(ascending=Falase)
df.sort_index()

# index:
df.reset_index(drop=True)
df.set_index('month')

# aggragtes | groupby
df.groupby("column_a")["column_b"].count()
df.groupby(["column_a", "column_b"])["column_c"].count()
df.groupby("column_a", sort= True)["column_b"].count()

df.groupby("column_a", sort= True)["column_b"].count().rename_axis(['column_a'])
df.groupby("column_a", sort= False)["column_b"].agg(['min', 'max'])

# Pivot


# lambda and apply_function:



