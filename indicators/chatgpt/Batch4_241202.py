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
    types.int64[:],  # on_qty_remain
    types.float32[:, :]  # curr_dataset
))
def OrderAmount(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    OrderAmount 因子计算函数：统计挂单金额。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    bid_idx = on_side == 0
    if np.any(bid_idx):  # 如果筛选到有效数据
        curr_dataset[0, 0] = np.sum((on_px[bid_idx] / 10000) * on_qty_remain[bid_idx])
    else:
        curr_dataset[0, 0] = 0  # 没有挂单金额则记为0

    # Ask侧计算
    ask_idx = on_side == 1
    if np.any(ask_idx):  # 如果筛选到有效数据
        curr_dataset[0, 1] = np.sum((on_px[ask_idx] / 10000) * on_qty_remain[ask_idx])
    else:
        curr_dataset[0, 1] = 0  # 没有挂单金额则记为0
        

@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # value_thresholds
    types.float32[:, :]  # curr_dataset
))
def LargeOrderAmountByValue(best_px, on_side, on_px, on_qty_org, on_qty_remain, value_thresholds, curr_dataset):
    """
    LargeOrderAmountByValue 因子计算函数：统计满足金额阈值的大单挂单金额总量。
    - value_thresholds：大单金额阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for T in value_thresholds:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px * on_qty_org / 10000 >= T)
        if np.any(bid_idx):  # 如果有满足条件的数据
            curr_dataset[index, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000)
        else:
            curr_dataset[index, 0] = 0  # 没有挂单金额则记为0

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px * on_qty_org / 10000 >= T)
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
    types.float64[:],  # value_thresholds
    types.float32[:, :]  # curr_dataset
))
def LargeOrderProportionByAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, value_thresholds, curr_dataset):
    """
    LargeOrderProportionByAmount 因子计算函数：统计满足金额阈值的大单挂单金额占比。
    - value_thresholds：大单金额阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for T in value_thresholds:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px * on_qty_org / 10000 >= T)
        total_bid_amount = np.sum(on_px[on_side == 0] * on_qty_remain[on_side == 0] / 10000)
        if total_bid_amount > 0:
            curr_dataset[index, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000) / total_bid_amount
        else:
            curr_dataset[index, 0] = 0

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px * on_qty_org / 10000 >= T)
        total_ask_amount = np.sum(on_px[on_side == 1] * on_qty_remain[on_side == 1] / 10000)
        if total_ask_amount > 0:
            curr_dataset[index, 1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx] / 10000) / total_ask_amount
        else:
            curr_dataset[index, 1] = 0

        index += 1