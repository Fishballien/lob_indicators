# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:25:48 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
from numba import typed, types
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


from utils.timeutils import get_a_share_intraday_time_series


# %%
symbol = '600519.XSHG'
data_dir = Path(r'D:\CNIndexFutures\timeseries\lob_indicators\sample_data')


# %% read data
trade_path = data_dir / 'trade' / f'{symbol}.parquet'
order_path = data_dir / 'order' / f'{symbol}.parquet'
trade_data = pd.read_parquet(trade_path)
order_data = pd.read_parquet(order_path)


# %%
start = datetime.now()


# %% preprocess
# 确保datetime列是datetime格式
trade_data['datetime'] = pd.to_datetime(trade_data['datetime'])

# 设置datetime为索引，用于重采样
trade_data = trade_data.set_index('datetime')

# 计算价格权重 (价格 * 成交量)
trade_data['weighted_price'] = trade_data['tradp'] * trade_data['tradv']

# 按3秒间隔重采样，右对齐
# 计算加权价格总和和成交量总和
total_weighted_price_3s = trade_data['weighted_price'].resample('3s', closed='right', label='right').sum()
total_volume_3s = trade_data['tradv'].resample('3s', closed='right', label='right').sum()

# 计算3秒间隔的TWAP
# 计算3秒间隔的TWAP，避免除以零
twap_3s = pd.Series(index=total_weighted_price_3s.index)
mask = total_volume_3s > 0
twap_3s[mask] = total_weighted_price_3s[mask] / total_volume_3s[mask] / 10000

# 处理可能的NaN值（没有交易的时间段）
twap_3s = twap_3s.ffill()

# 现在按1分钟间隔重采样，右对齐
# 对于分钟TWAP，我们取3秒TWAP的平均值
minute_twap = twap_3s.resample('1min', closed='right', label='right').mean()

# 显示结果
print(minute_twap.head())


date_in_dt = datetime(2024, 10, 29)
target_ts_param = {'minutes': 1}
target_ts = get_a_share_intraday_time_series(date_in_dt, target_ts_param).view('M8[ms]')
minute_twap = minute_twap.reindex(target_ts)

res_descr =  [('timestamp', 'i8'), ('twap_1min', 'f8')]
res_dtype = np.dtype(res_descr)
res = np.full(len(target_ts), fill_value=np.nan, dtype=res_dtype)
res['timestamp'] = target_ts
res['twap_1min'] = minute_twap.values


# %%
end = datetime.now()
print(end-start)




























