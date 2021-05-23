import pandas as pd 

s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd', 'c'])

print(s1), print(s2)

#adding them row wise
s3 = pd.concat([s1, s2], axis = 0, ignore_index = True)
print(s3)

#adding then column wise
s3 = pd.concat([s1, s2], axis = 1, ignore_index = True)
print(s3)

#inner join
df1 = pd.DataFrame([['a', 1], ['b', 2]], columns = ['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns = ['letter', 'number'])
df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns = ['letter', 'number', 'animal'])

df4 = pd.concat([df1,  df3], join = 'inner')
print(df4)

df5 = pd.concat([df1,  df3], join = 'outer')
print(df5)

df6 = pd.concat([df1, df2, df3], join = 'inner', ignore_index = True)
print(df6)

df7 = pd.concat([df1, df2, df3], join = 'outer', ignore_index = True)
print(df7)

df8 = pd.DataFrame([1], index = ['a'])
print(df8)

df9 = pd.DataFrame([2], index = ['b'])
print(df9)

df10 = pd.concat([df8, df9], verify_integrity = True)
print(df10)

df11 = pd.DataFrame([3], index = ['b'])
print(df9)

#verify_integrity throws valueError becuase of duplicate indexes
df12 = pd.concat([df9, df11], verify_integrity = True)
print(df10) 

#hierarchical index
s4 = pd.concat([s1, s2], keys = ['s1', 's2'])
print(s4)
 
s5 = pd.concat([s1, s2], key = ['s1', 's2'], names = ['Series name', 'Row ID'])
print(s5)
