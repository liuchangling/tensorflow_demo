将excel打乱
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html#pandas.DataFrame.to_excel

import pandas as pd
df = pd.read_excel('data.xlsx')
df = shuffle(df)
df.to_excel("output.xlsx")  