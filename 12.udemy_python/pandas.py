import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import webbrowser

website = 'http://en.wikipedia.org/wiki/NFL_win-loss_records'
webbrowser.open(website)


# クリップボードから読み込むことが可能です。
nfl_frame = pd.read_clipboard()

nfl_frame.head(4)
nfl_frame.tail()
nfl_frame.columns
nfl_frame['Team']

DataFrame(nfl_frame,  columns=['Team', 'Division'])

# 新しいカラムを追加
nfl_frame['Stadium'] = "Levis Stadium"
nfl_frame['Stadium'] = np.arange(21)

# Seriesを作成して、新しいカラムを作成
stadiums = Series(["Levis Stadium", "AT&T Stadium"], [1,0])
nfl_frame['Stadium'] = stadiums

# カラムの削除
del nfl_frame['Stadium']


# DataFrameの作成
dframe = DataFrame(np.arange(25).reshape((5,5)),
    index=['NYC','LA','SF','DC','Chi'],
    columns=['A','B','C','D','E'])

dframe['B']
dframe[['B', 'C']]

dframe[dframe['C']>8]

arr = np.array([[1,2,np.nan], [np.nan, 3,4]])

dframe1 = DataFrame(arr, 
    index=['A', 'B'],
    columns=['one', 'two', 'three'])

dframe1

dframe1.describe()



# 株価のデータを使って、共分散と相関をみていきましょう。
# PandasはWebからデータをとってくることも可能です。
import pandas_datareader as pdweb
# 日付を扱うために、datetimeをimportします。
import datetime
# 米国のYahooのサービスを使って、株価を取得します。
# すべて石油関連会社
# CVX シェブロン、XOM エクソンモービル、BP 英BP
prices = pdweb.get_data_yahoo(['CVX','XOM','BP'], 
                               start=datetime.datetime(2010, 1, 1), 
                               end=datetime.datetime(2013, 1, 1))['Adj Close']
prices.head(10)


rets = prices.pct_change()
rets.head()


%matplotlib inline
prices.plot()

# 時系列データの相関？を算出する
rets.corr()


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(rets.corr())






















