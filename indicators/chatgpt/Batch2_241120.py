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
@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def PriceCenterAbsDeviation(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：重心价格偏离中间价距离
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx]
            weighted_center = safe_divide(np.sum(on_px[bid_idx] * weights), np.sum(weights))
            curr_dataset[index, 0] = abs(weighted_center - mid_price)
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx]
            weighted_center = safe_divide(np.sum(on_px[ask_idx] * weights), np.sum(weights))
            curr_dataset[index, 1] = abs(weighted_center - mid_price)
        else:
            curr_dataset[index, 1] = np.nan

        index += 1
        
@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def Momentum(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    因子：价格动能
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧
    bid_idx = (on_side == 0) & (on_px <= bid1) & (on_qty_remain > 0)
    if np.any(bid_idx):
        curr_dataset[0, 0] = np.sum((on_px[bid_idx] - mid_price) * on_qty_remain[bid_idx])
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧
    ask_idx = (on_side == 1) & (on_px >= ask1) & (on_qty_remain > 0)
    if np.any(ask_idx):
        curr_dataset[0, 1] = np.sum((on_px[ask_idx] - mid_price) * on_qty_remain[ask_idx])
    else:
        curr_dataset[0, 1] = np.nan
        
@timeit       
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def Inertia(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    因子：价格转动惯量
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧
    bid_idx = (on_side == 0) & (on_px <= bid1) & (on_qty_remain > 0)
    if np.any(bid_idx):
        curr_dataset[0, 0] = np.sum(((on_px[bid_idx] - mid_price) ** 2) * on_qty_remain[bid_idx])
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧
    ask_idx = (on_side == 1) & (on_px >= ask1) & (on_qty_remain > 0)
    if np.any(ask_idx):
        curr_dataset[0, 1] = np.sum(((on_px[ask_idx] - mid_price) ** 2) * on_qty_remain[ask_idx])
    else:
        curr_dataset[0, 1] = np.nan

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def PriceEntropy(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单价格分布的熵
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            qty_sum = np.sum(on_qty_remain[bid_idx])
            p = on_qty_remain[bid_idx] / qty_sum
            entropy = -np.sum(p * np.log(p))
            curr_dataset[index, 0] = entropy
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            qty_sum = np.sum(on_qty_remain[ask_idx])
            p = on_qty_remain[ask_idx] / qty_sum
            entropy = -np.sum(p * np.log(p))
            curr_dataset[index, 1] = entropy
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def RemainingOrderSkewness(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：剩余挂单价格分布的偏度
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            mean_px = safe_divide(np.sum(prices * weights), np.sum(weights))
            diff = prices - mean_px
            skewness = safe_divide(np.sum(weights * diff**3), (np.sum(weights * diff**2)**1.5))
            curr_dataset[index, 0] = skewness
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            mean_px = safe_divide(np.sum(prices * weights), np.sum(weights))
            diff = prices - mean_px
            skewness = safe_divide(np.sum(weights * diff**3) , (np.sum(weights * diff**2)**1.5))
            curr_dataset[index, 1] = skewness
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def RemainingOrderKurtosis(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：剩余挂单价格分布的峰度
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            mean_px = safe_divide(np.sum(prices * weights), np.sum(weights))
            diff = prices - mean_px
            kurtosis = safe_divide(np.sum(weights * diff**4), (np.sum(weights * diff**2)**2))
            curr_dataset[index, 0] = kurtosis
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            mean_px = safe_divide(np.sum(prices * weights), np.sum(weights))
            diff = prices - mean_px
            kurtosis = safe_divide(np.sum(weights * diff**4), (np.sum(weights * diff**2)**2))
            curr_dataset[index, 1] = kurtosis
        else:
            curr_dataset[index, 1] = np.nan

        index += 1
        
@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # local_range
    types.float64[:, :]  # curr_dataset
))
def LocalPriceEntropy(best_px, on_side, on_px, on_qty_remain, local_range, curr_dataset):
    """
    因子：挂单金额分布的局部熵
    - local_range: 局部范围的价格区间宽度（相对中间价的百分比）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for lr in local_range:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= bid1 * (1 - lr)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx]
            probs = weights / np.sum(weights)
            local_entropy = -np.sum(probs * np.log(probs))
            curr_dataset[index, 0] = local_entropy
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= ask1 * (1 + lr)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx]
            probs = weights / np.sum(weights)
            local_entropy = -np.sum(probs * np.log(probs))
            curr_dataset[index, 1] = local_entropy
        else:
            curr_dataset[index, 1] = np.nan

        index += 1
        
@timeit      
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def GiniCoefficient(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单金额分布的Gini系数
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            # 按价格升序排序
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            n = len(sorted_weights)
            cumulative_weights = np.cumsum(sorted_weights)
            gini = 1 - (2 * np.sum((np.arange(1, n + 1) * sorted_weights)) / (n * np.sum(sorted_weights))) + 1 / n
            curr_dataset[index, 0] = gini
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            # 按价格升序排序
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            n = len(sorted_weights)
            cumulative_weights = np.cumsum(sorted_weights)
            gini = 1 - (2 * np.sum((np.arange(1, n + 1) * sorted_weights)) / (n * np.sum(sorted_weights))) + 1 / n
            curr_dataset[index, 1] = gini
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # local_range
    types.float64[:, :]  # curr_dataset
))
def LocalGiniCoefficient(best_px, on_side, on_px, on_qty_remain, local_range, curr_dataset):
    """
    因子：挂单价格区间的局部Gini系数
    - local_range: 局部价格区间宽度（相对中间价的百分比）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for lr in local_range:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= bid1 * (1 - lr)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            n = len(sorted_weights)
            gini = 1 - (2 * np.sum((np.arange(1, n + 1) * sorted_weights)) / (n * np.sum(sorted_weights))) + 1 / n
            curr_dataset[index, 0] = gini
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= ask1 * (1 + lr)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            n = len(sorted_weights)
            gini = 1 - (2 * np.sum((np.arange(1, n + 1) * sorted_weights)) / (n * np.sum(sorted_weights))) + 1 / n
            curr_dataset[index, 1] = gini
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def WeightedPriceStd(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单价格分布的加权标准差
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            mean_price = np.sum(prices * weights) / np.sum(weights)
            weighted_std = np.sqrt(np.sum(weights * (prices - mean_price)**2) / np.sum(weights))
            curr_dataset[index, 0] = weighted_std
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            mean_price = np.sum(prices * weights) / np.sum(weights)
            weighted_std = np.sqrt(np.sum(weights * (prices - mean_price)**2) / np.sum(weights))
            curr_dataset[index, 1] = weighted_std
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def WeightedPriceMAD(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单价格分布的加权平均绝对偏差
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            mean_price = np.sum(prices * weights) / np.sum(weights)
            weighted_mad = np.sum(weights * np.abs(prices - mean_price)) / np.sum(weights)
            curr_dataset[index, 0] = weighted_mad
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            mean_price = np.sum(prices * weights) / np.sum(weights)
            weighted_mad = np.sum(weights * np.abs(prices - mean_price)) / np.sum(weights)
            curr_dataset[index, 1] = weighted_mad
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def MaxOrderRatio(best_px, on_side, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：最大金额挂单的占比
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_qty_remain > 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx]
            max_ratio = np.max(weights) / np.sum(weights)
            curr_dataset[index, 0] = max_ratio
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_qty_remain > 0)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx]
            max_ratio = np.max(weights) / np.sum(weights)
            curr_dataset[index, 1] = max_ratio
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def OrderBookSecondDiff(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：订单簿的二阶差分
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            # 按价格升序排序
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            second_diff = np.diff(sorted_weights, n=2)
            curr_dataset[index, 0] = np.mean(second_diff)  # 取二阶差分的平均值
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            second_diff = np.diff(sorted_weights, n=2)
            curr_dataset[index, 1] = np.mean(second_diff)
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def PriceQtyCorrelation(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：价格与挂单金额的相关性
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            mean_price = np.mean(prices)
            mean_weight = np.mean(weights)
            covariance = np.sum((prices - mean_price) * (weights - mean_weight))
            variance_price = np.sum((prices - mean_price)**2)
            variance_weight = np.sum((weights - mean_weight)**2)
            correlation = safe_divide(covariance, np.sqrt(variance_price * variance_weight))
            curr_dataset[index, 0] = correlation
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            mean_price = np.mean(prices)
            mean_weight = np.mean(weights)
            covariance = np.sum((prices - mean_price) * (weights - mean_weight))
            variance_price = np.sum((prices - mean_price)**2)
            variance_weight = np.sum((weights - mean_weight)**2)
            correlation = safe_divide(covariance, np.sqrt(variance_price * variance_weight))
            curr_dataset[index, 1] = correlation
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def HerfindahlIndex(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单金额的Herfindahl指数
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx]
            proportions = weights / np.sum(weights)
            herfindahl = np.sum(proportions**2)
            curr_dataset[index, 0] = herfindahl
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx]
            proportions = weights / np.sum(weights)
            herfindahl = np.sum(proportions**2)
            curr_dataset[index, 1] = herfindahl
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

@timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_threshold
    types.float64[:, :]  # curr_dataset
))
def Smoothness(best_px, on_side, on_px, on_qty_remain, price_threshold, curr_dataset):
    """
    因子：挂单金额分布的光滑度
    - price_threshold: 相对中间价的筛选范围
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_price = (bid1 + ask1) / 2

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for threshold in price_threshold:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px <= bid1) & (on_px >= mid_price * (1 - threshold)) & (on_qty_remain > 0)
        if np.any(bid_idx):
            prices = on_px[bid_idx]
            weights = on_qty_remain[bid_idx]
            # 按价格升序排序
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            smoothness = -np.sum(np.abs(np.diff(sorted_weights)))  # 计算光滑度
            curr_dataset[index, 0] = smoothness
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1) & (on_px <= mid_price * (1 + threshold)) & (on_qty_remain > 0)
        if np.any(ask_idx):
            prices = on_px[ask_idx]
            weights = on_qty_remain[ask_idx]
            sorted_idx = np.argsort(prices)
            sorted_weights = weights[sorted_idx]
            smoothness = -np.sum(np.abs(np.diff(sorted_weights)))  # 计算光滑度
            curr_dataset[index, 1] = smoothness
        else:
            curr_dataset[index, 1] = np.nan

        index += 1
