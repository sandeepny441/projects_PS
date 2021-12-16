#chapter: 1
df = pd.read_csv()
df = pd.read_html()
df = pd.read_excel()

df.index
df.columns
df.values
df.dtypes 

issubclass(chidl_onject, parent_object)

s.to_frame()
s.shape
s.size
s.min()
s.max()
s.mean()
s.quantile(0.3)


df['A'].value_counts()
df['A'].value_counts(normalize = True)
df['A'].count()

df['A'].min()
df['A'].max()
df['A'].mean()
df['A'].quantile(0.3)

df['A'].isnull()
df['A'].isna()
df['A'].isnull().sum()
df['A'].isnull().sum().sum()
df['A'].dropna()


df.shape 
df.size 






