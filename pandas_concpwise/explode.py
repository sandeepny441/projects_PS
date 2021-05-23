import pandas as pd


#general_explode
df = pd.DataFrame({'A': [[1,3],[3]], 'B': 1})
print("------The DataFrame is--------")
print(df)
print("After expanding the DataFrame")
print(df.explode('A'))

#reset_index
df = pd.DataFrame({'A': [[1,3],[4]], 'B': 1})
print("------The DataFrame is--------")
print(df)
print("After expanding the DataFrame")
print(df.explode('A').reset_index(drop = True))

#The empty lists will be expanded into a numpy.nan value.
df = pd.DataFrame({'A': [[1, 2], []], 'B': 1})
print("------The DataFrame is--------")
print(df)
print("After expanding the DataFrame")
print(df.explode('A').reset_index(drop = True))

