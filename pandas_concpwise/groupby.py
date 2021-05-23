import pandas as pd 
# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                               'Parrot', 'Parrot'],
#                    'Max Speed': [380., 370., 24., 26.]})

# print(df)
data = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
df = pd.DataFrame(data, columns=["a", "b", "c"])
print(df)

print('==============================================')
df1= df.groupby(by=["a"]).sum()
