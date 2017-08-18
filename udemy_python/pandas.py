import numpy as np
import pandas as pd
from pandas import Series, DataFrame
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
