# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:33:36 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
from datetime import datetime


from utils.timeutils import get_a_share_intraday_time_series


# %%
class FixedTimeIntervalLoop:
    
    date_format_list = ['%Y%m%d', '%Y-%m-%d']
    
    def __init__(self, date, loop_func, target_ts_param, len_of_data):
        self.loop_func = loop_func
        self.len_of_data = len_of_data
        
        date_in_dt = None
        for dt_format in self.date_format_list:
            try:
                date_in_dt = datetime.strptime(date, dt_format)
                break
            except:
                pass
        self.target_ts = get_a_share_intraday_time_series(date_in_dt, target_ts_param)
        self.len_of_ts = len(self.target_ts)
        self.start_idx = 0
        self.ts_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx >= self.len_of_data or self.ts_idx >= self.len_of_ts:
            # print(self.start_idx, self.len_of_data, self.ts_idx, self.len_of_ts)
            raise StopIteration
        
        nxt_target_ts = self.target_ts[self.ts_idx]
        self.start_idx = self.loop_func(self.start_idx, nxt_target_ts)
        ts_idx = self.ts_idx
        self.ts_idx += 1
        return ts_idx, nxt_target_ts