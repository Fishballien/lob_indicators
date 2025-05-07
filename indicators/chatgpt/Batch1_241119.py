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
from numba import njit, types, prange


from utils.assist_calc import get_residue_time
from utils.speedutils import timeit


# %%
@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_adjustment
    types.float64[:],  # price_range
    types.float64[:, :]  # curr_dataset
))
def AdjustedVolumeDistribution(best_px, on_side, on_px, on_qty_remain, price_adjustment, price_range, curr_dataset):
    """
    挂单价差调整后的量能分布计算
    - price_adjustment: 调整价格的比例
    - price_range: 调整后价格筛选的上下限
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历参数组合
    for adj in price_adjustment:
        for pr in price_range:
            P_min = bid1 * (1 - pr)
            P_max = ask1 * (1 + pr)

            # Bid侧计算
            adjusted_bid_px = on_px * (1 + adj)
            bid_idx = (on_side == 0) & (adjusted_bid_px <= bid1) & (adjusted_bid_px >= P_min)
            if np.any(bid_idx):
                curr_dataset[index, 0] = np.sum(on_qty_remain[bid_idx])
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            adjusted_ask_px = on_px * (1 + adj)
            ask_idx = (on_side == 1) & (adjusted_ask_px >= ask1) & (adjusted_ask_px <= P_max)
            if np.any(ask_idx):
                curr_dataset[index, 1] = np.sum(on_qty_remain[ask_idx])
            else:
                curr_dataset[index, 1] = np.nan

            # 更新行索引
            index += 1
            

@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderBookEntropy(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单熵计算
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧熵计算
    bid_idx = (on_side == 0) & (on_qty_remain > 0)
    if np.any(bid_idx):
        bid_remain = on_qty_remain[bid_idx]
        p = bid_remain / np.sum(bid_remain)
        entropy = -np.sum(p * np.log(p))
        curr_dataset[0, 0] = entropy
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧熵计算
    ask_idx = (on_side == 1) & (on_qty_remain > 0)
    if np.any(ask_idx):
        ask_remain = on_qty_remain[ask_idx]
        p = ask_remain / np.sum(ask_remain)
        entropy = -np.sum(p * np.log(p))
        curr_dataset[0, 1] = entropy
    else:
        curr_dataset[0, 1] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def AccelerationOrderVolume(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    加速度挂单量计算
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    P_mid = (bid1 + ask1) / 2

    # Bid侧加速度计算
    bid_idx = (on_side == 0) & (on_qty_remain > 0)
    if np.any(bid_idx):
        bid_remain = on_qty_remain[bid_idx]
        bid_px_diff = (on_px[bid_idx] - P_mid) ** 2
        curr_dataset[0, 0] = np.sum(bid_px_diff * bid_remain) / np.sum(bid_remain)
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧加速度计算
    ask_idx = (on_side == 1) & (on_qty_remain > 0)
    if np.any(ask_idx):
        ask_remain = on_qty_remain[ask_idx]
        ask_px_diff = (on_px[ask_idx] - P_mid) ** 2
        curr_dataset[0, 1] = np.sum(ask_px_diff * ask_remain) / np.sum(ask_remain)
    else:
        curr_dataset[0, 1] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def LargeOrderRemaining(best_px, on_side, on_px, on_qty_org, on_qty_remain, amount_threshold, curr_dataset):
    """
    大单挂单剩余量计算
    - amount_threshold: 大单金额的筛选阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历阈值
    for threshold in amount_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_qty_org * on_px / 10000 >= threshold)
        if np.any(bid_idx):
            curr_dataset[index, 0] = np.sum(on_qty_remain[bid_idx])
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_qty_org * on_px / 10000 >= threshold)
        if np.any(ask_idx):
            curr_dataset[index, 1] = np.sum(on_qty_remain[ask_idx])
        else:
            curr_dataset[index, 1] = np.nan

        # 更新行索引
        index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.int64[:],  # on_qty_remain
    types.float64[:],  # min_weight
    types.float64[:],  # decay_rate
    types.float64[:, :]  # curr_dataset
))
def TimeWeightedRemainingVolume(best_px, on_side, on_ts_org, ts, on_qty_remain, min_weight, decay_rate, curr_dataset):
    """
    挂单时间加权剩余量计算
    - min_weight: 权重最小值 (a)
    - decay_rate: 衰减速度参数 (b)
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 计算当前挂单时间，剔除非交易时段
    time_diff = get_residue_time(ts, on_ts_org)  # 剩余时间函数，需外部实现

    # 初始化行索引
    index = 0

    # 遍历参数组合
    for a in min_weight:
        for b in decay_rate:
            # 权重公式：w(t) = a + (1 - a) * exp(-b * t)
            weight = a + (1 - a) * np.exp(-b * time_diff)

            # Bid侧计算
            bid_idx = (on_side == 0) & (time_diff > 0)
            if np.any(bid_idx):
                weighted_bid_vol = np.sum(weight[bid_idx] * on_qty_remain[bid_idx])
                curr_dataset[index, 0] = weighted_bid_vol
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (time_diff > 0)
            if np.any(ask_idx):
                weighted_ask_vol = np.sum(weight[ask_idx] * on_qty_remain[ask_idx])
                curr_dataset[index, 1] = weighted_ask_vol
            else:
                curr_dataset[index, 1] = np.nan

            # 更新行索引
            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_span
    types.float64[:, :]  # curr_dataset
))
def ClusteringOrderDensity(best_px, on_side, on_px, on_qty_remain, price_span, curr_dataset):
    """
    聚集性挂单密度计算
    - price_span: 价格范围的相对跨度
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    P_mid = (bid1 + ask1) / 2

    # 初始化行索引
    index = 0

    for delta in price_span:
        P_low = P_mid * (1 - delta)
        P_high = P_mid * (1 + delta)

        # Bid侧密度
        bid_idx = (on_side == 0) & (on_px >= P_low) & (on_px <= P_mid)
        if np.any(bid_idx):
            bid_density = np.sum(on_qty_remain[bid_idx]) / (bid1 - P_low)
            curr_dataset[index, 0] = bid_density
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧密度
        ask_idx = (on_side == 1) & (on_px >= P_mid) & (on_px <= P_high)
        if np.any(ask_idx):
            ask_density = np.sum(on_qty_remain[ask_idx]) / (P_high - ask1)
            curr_dataset[index, 1] = ask_density
        else:
            curr_dataset[index, 1] = np.nan

        index += 1
