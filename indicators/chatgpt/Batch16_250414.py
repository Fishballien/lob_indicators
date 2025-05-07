# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:27:51 2024

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


from utils.assist_calc import get_residue_time, safe_divide, safe_divide_arrays, safe_divide_array_by_scalar
from utils.speedutils import timeit


# %%
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def TotalOrderAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, curr_dataset):
    """
    TotalOrderAmount 因子计算函数：统计所有挂单金额总量，不区分大小单。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # Bid侧计算 - 所有bid侧挂单总金额
    bid_idx = (on_side == 0)
    if np.any(bid_idx):  # 如果有满足条件的数据
        curr_dataset[0, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000)
    else:
        curr_dataset[0, 0] = 0  # 没有挂单金额则记为0
    
    # Ask侧计算 - 所有ask侧挂单总金额
    ask_idx = (on_side == 1)
    if np.any(ask_idx):  # 如果有满足条件的数据
        curr_dataset[0, 1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx] / 10000)
    else:
        curr_dataset[0, 1] = 0  # 没有挂单金额则记为0


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # value_thresholds
    types.float64[:, :]  # curr_dataset
))
def SmallOrderAmountByValue(best_px, on_side, on_px, on_qty_org, on_qty_remain, value_thresholds, curr_dataset):
    """
    SmallOrderAmountByValue 因子计算函数：统计小于金额阈值的小单挂单金额总量。
    - value_thresholds：小单金额阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    index = 0
    for T in value_thresholds:
        # Bid侧计算 - 小于阈值T的bid侧小单总金额
        bid_idx = (on_side == 0) & (on_px * on_qty_org / 10000 < T)
        if np.any(bid_idx):  # 如果有满足条件的数据
            curr_dataset[index, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000)
        else:
            curr_dataset[index, 0] = 0  # 没有挂单金额则记为0
        
        # Ask侧计算 - 小于阈值T的ask侧小单总金额
        ask_idx = (on_side == 1) & (on_px * on_qty_org / 10000 < T)
        if np.any(ask_idx):  # 如果有满足条件的数据
            curr_dataset[index, 1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx] / 10000)
        else:
            curr_dataset[index, 1] = 0  # 没有挂单金额则记为0
        
        index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # percent_thresholds
    types.float64[:, :]  # curr_dataset
))
def OrderAmountOutsidePriceRange(best_px, on_side, on_px, on_qty_org, on_qty_remain, percent_thresholds, curr_dataset):
    """
    OrderAmountOutsidePriceRange 因子计算函数：统计中间价格上下一定百分比范围外的挂单金额总量。
    - percent_thresholds：价格偏离中间价的百分比阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 计算中间价
    mid_price = (bid1 + ask1) / 2
    
    index = 0
    for percent in percent_thresholds:
        # 计算价格范围
        lower_bound = mid_price * (1 - percent)
        upper_bound = mid_price * (1 + percent)
        
        # Bid侧计算 - 价格低于lower_bound的bid侧挂单总金额
        bid_idx = (on_side == 0) & (on_px < lower_bound)
        if np.any(bid_idx):  # 如果有满足条件的数据
            curr_dataset[index, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000)
        else:
            curr_dataset[index, 0] = 0  # 没有挂单金额则记为0
        
        # Ask侧计算 - 价格高于upper_bound的ask侧挂单总金额
        ask_idx = (on_side == 1) & (on_px > upper_bound)
        if np.any(ask_idx):  # 如果有满足条件的数据
            curr_dataset[index, 1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx] / 10000)
        else:
            curr_dataset[index, 1] = 0  # 没有挂单金额则记为0
        
        index += 1