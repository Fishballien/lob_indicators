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
from numba import njit, types
from functools import partial


# %% import self_defined
from core.go_through_book import GoThroughBookStepper
from utils.market import PX_MULTIPLIER


# %%
class Indv0(GoThroughBookStepper):
    
    def _init_curr_dataset(self):
        self.curr_dataset = np.zeros(len(self.recorded_dtype), dtype=np.float64)
    
    def _init_indicator_dtype(self):
        return np.dtype([ 
            ('bid_total_amount', np.int64), ('ask_total_amount', np.int64),
            ('bid_l_amount', np.int64), ('ask_l_amount', np.int64),
            ])
    
    def run(self):
        l_thres = self.param['l_thres']
        update_static_indicators_func = partial(update_static_indicators_wrapper,
                                                best_px=self.best_px_post_match, on_side=self.on_side, on_px=self.on_px, 
                                                on_qty_org=self.on_qty_org, on_qty_remain=self.on_qty_remain, 
                                                l_thres=l_thres, curr_dataset=self.curr_dataset)
        for ts_idx in self.stepper:
            update_static_indicators_func()
            for idx, name in enumerate(self.recorded_dtype.names):
                self.recorded_dataset[ts_idx][name] = self.curr_dataset[idx]
        res = self.final()
        return res
    
    
# %%
def update_static_indicators_wrapper(best_px, on_side, on_px, on_qty_org, on_qty_remain, l_thres, curr_dataset):
    update_static_indicators(best_px, on_side, on_px, on_qty_org, on_qty_remain, l_thres, curr_dataset)
    

@njit(types.void(types.int64[:], types.int32[:], types.int64[:], 
                  types.int64[:], types.int64[:], types.int64, types.float64[:]))     
def update_static_indicators(best_px, on_side, on_px, on_qty_org, on_qty_remain, l_thres, curr_dataset):
    
    bid1 = best_px[0]
    ask1 = best_px[1]
    l_idx = (on_qty_org * on_px) / PX_MULTIPLIER >= l_thres
    
    bid_idx = (on_side == 0) & (on_px <= bid1)
    curr_dataset[0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx]) / PX_MULTIPLIER
    curr_dataset[2] = np.sum(on_px[bid_idx&l_idx] * on_qty_remain[bid_idx&l_idx]) / PX_MULTIPLIER

    ask_idx = (on_side == 1) & (on_px >= ask1)
    curr_dataset[1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx]) / PX_MULTIPLIER
    curr_dataset[3] = np.sum(on_px[ask_idx&l_idx] * on_qty_remain[ask_idx&l_idx]) / PX_MULTIPLIER
