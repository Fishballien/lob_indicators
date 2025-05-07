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
    types.int64[:],  # on_ts_org
    types.int64,  # ts
    types.float64[:],  # time_thresholds (秒)
    types.float32[:, :]  # curr_dataset
))
def RecentOrderAmountVolatility(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, time_thresholds, curr_dataset):
    """
    计算Bid和Ask的近期挂单金额波动率因子：根据多个时间阈值分别计算Bid和Ask的近期挂单金额波动性。
    - time_thresholds: 用于计算"近期"定义的时间阈值（秒）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    time_diff = ts - on_ts_org  # 计算时间差

    for i, time_threshold in enumerate(time_thresholds):
        time_threshold_ms = time_threshold * 1000  # 秒转毫秒

        # 筛选出Bid和Ask侧"近期挂单"的索引
        recent_idx = time_diff <= time_threshold_ms

        # Bid侧的金额
        bid_amounts = on_px[recent_idx & (on_side == 0)] * on_qty_remain[recent_idx & (on_side == 0)] / 10000  # 除以10000得到实际金额
        if len(bid_amounts) > 0:
            bid_mean_amount = np.mean(bid_amounts)
            bid_std_amount = np.std(bid_amounts)
            if bid_mean_amount != 0:
                curr_dataset[i, 0] = bid_std_amount / bid_mean_amount  # 存储Bid侧波动率
            else:
                curr_dataset[i, 0] = np.nan
        else:
            curr_dataset[i, 0] = np.nan

        # Ask侧的金额
        ask_amounts = on_px[recent_idx & (on_side == 1)] * on_qty_remain[recent_idx & (on_side == 1)] / 10000  # 除以10000得到实际金额
        if len(ask_amounts) > 0:
            ask_mean_amount = np.mean(ask_amounts)
            ask_std_amount = np.std(ask_amounts)
            if ask_mean_amount != 0:
                curr_dataset[i, 1] = ask_std_amount / ask_mean_amount  # 存储Ask侧波动率
            else:
                curr_dataset[i, 1] = np.nan
        else:
            curr_dataset[i, 1] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,  # ts
    types.float64[:],  # time_thresholds (秒)
    types.float32[:, :]  # curr_dataset
))
def RecentOrderAmountCumulative(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, time_thresholds, curr_dataset):
    """
    计算Bid和Ask的近期挂单金额累计总量因子：计算"近期"挂单金额的累计总和。
    - time_thresholds: 用于计算"近期"定义的时间阈值（秒）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    time_diff = ts - on_ts_org  # 计算时间差

    for i, time_threshold in enumerate(time_thresholds):
        time_threshold_ms = time_threshold * 1000  # 秒转毫秒

        # 筛选出Bid和Ask侧"近期挂单"的索引
        recent_idx = time_diff <= time_threshold_ms

        # Bid侧的金额
        bid_amount = np.sum(on_px[recent_idx & (on_side == 0)] * on_qty_remain[recent_idx & (on_side == 0)] / 10000)  # 除以10000得到实际金额
        curr_dataset[i, 0] = bid_amount

        # Ask侧的金额
        ask_amount = np.sum(on_px[recent_idx & (on_side == 1)] * on_qty_remain[recent_idx & (on_side == 1)] / 10000)  # 除以10000得到实际金额
        curr_dataset[i, 1] = ask_amount


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,  # ts
    types.float64[:],  # time_thresholds (秒)
    types.float32[:, :]  # curr_dataset
))
def RecentOrderAmountRatio(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, time_thresholds, curr_dataset):
    """
    计算Bid和Ask的近期挂单金额占比因子：计算"近期"挂单金额占比。
    - time_thresholds: 用于计算"近期"定义的时间阈值（秒）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    time_diff = ts - on_ts_org  # 计算时间差

    for i, time_threshold in enumerate(time_thresholds):
        time_threshold_ms = time_threshold * 1000  # 秒转毫秒

        # 筛选出Bid和Ask侧"近期挂单"的索引
        recent_idx = time_diff <= time_threshold_ms

        # Bid侧的金额占比
        bid_total_amount = np.sum(on_px[(on_side == 0)] * on_qty_remain[(on_side == 0)] / 10000)
        bid_recent_amount = np.sum(on_px[recent_idx & (on_side == 0)] * on_qty_remain[recent_idx & (on_side == 0)] / 10000)  # 除以10000得到实际金额
        if bid_total_amount != 0:
            curr_dataset[i, 0] = bid_recent_amount / bid_total_amount  # 存储Bid侧金额占比
        else:
            curr_dataset[i, 0] = np.nan

        # Ask侧的金额占比
        ask_total_amount = np.sum(on_px[(on_side == 1)] * on_qty_remain[(on_side == 1)] / 10000)
        ask_recent_amount = np.sum(on_px[recent_idx & (on_side == 1)] * on_qty_remain[recent_idx & (on_side == 1)] / 10000)  # 除以10000得到实际金额
        if ask_total_amount != 0:
            curr_dataset[i, 1] = ask_recent_amount / ask_total_amount  # 存储Ask侧金额占比
        else:
            curr_dataset[i, 1] = np.nan

