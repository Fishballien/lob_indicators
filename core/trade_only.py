# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:44:17 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


from utils.timeutils import get_a_share_intraday_time_series


# %%
class ProcessTwap:
    
    date_format_list = ['%Y%m%d', '%Y-%m-%d']
    
    def __init__(self, symbol, date, order_data, trade_data, param):
        self.date = date
        self.param = param
        self.trade_data = trade_data
        
    def run(self):
        trade_data = self.trade_data
        target_ts_param = self.param['target_ts']
        for dt_format in self.date_format_list:
            try:
                date_in_dt = datetime.strptime(self.date, dt_format)
                break
            except:
                pass
        target_ts = get_a_share_intraday_time_series(date_in_dt, target_ts_param)
        
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
        twap_3s = total_weighted_price_3s / total_volume_3s / 10000

        # 处理可能的NaN值（没有交易的时间段）
        twap_3s = twap_3s.ffill()

        # 现在按1分钟间隔重采样，右对齐
        # 对于分钟TWAP，我们取3秒TWAP的平均值
        minute_twap = twap_3s.resample('1min', closed='right', label='right').mean()

        minute_twap = minute_twap.reindex(target_ts)
        
        res_descr =  [('timestamp', 'i8'), ('twap_1min', 'f8')]
        res_dtype = np.dtype(res_descr)
        res = np.full(len(target_ts), fill_value=np.nan, dtype=res_dtype)
        res['timestamp'] = target_ts
        res['twap_1min'] = minute_twap.values
        
        return res
        