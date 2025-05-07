# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:46:23 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np


# %% import self_defined
from core.go_through_book import GoThroughBookStepper


# %%
class PriceV0(GoThroughBookStepper):
    
    def _init_curr_dataset(self):
        self.curr_dataset = np.zeros(len(self.recorded_dtype), dtype=np.int64)
    
    def _init_indicator_dtype(self):
        return np.dtype([('bid1', np.int64), ('ask1', np.int64)])
    
    def run(self):
        for ts_idx in self.stepper:
            self.recorded_dataset[ts_idx]['bid1'] = self.best_px_post_match[0]
            self.recorded_dataset[ts_idx]['ask1'] = self.best_px_post_match[1]
        return self.final()