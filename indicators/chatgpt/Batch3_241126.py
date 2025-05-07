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
# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderVolume(best_px, on_side, on_qty_remain, curr_dataset):
    """
    挂单总量因子
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    bid_idx = (on_side == 0)
    curr_dataset[0, 0] = np.sum(on_qty_remain[bid_idx]) if np.any(bid_idx) else 0

    # Ask侧计算
    ask_idx = (on_side == 1)
    curr_dataset[0, 1] = np.sum(on_qty_remain[ask_idx]) if np.any(ask_idx) else 0

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # value_thresholds
    types.float64[:, :]  # curr_dataset
))
def LargeOrderVolumeByValue(best_px, on_side, on_px, on_qty_org, on_qty_remain, value_thresholds, curr_dataset):
    """
    大金额挂单量总量因子
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
        bid_idx = (on_side == 0) & (on_qty_org * on_px / 10000 >= T)
        curr_dataset[index, 0] = np.sum(on_qty_remain[bid_idx]) if np.any(bid_idx) else 0

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_qty_org * on_px / 10000 >= T)
        curr_dataset[index, 1] = np.sum(on_qty_remain[ask_idx]) if np.any(ask_idx) else 0

        index += 1

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # value_thresholds
    types.float64[:, :]  # curr_dataset
))
def LargeOrderProportionByValue(best_px, on_side, on_px, on_qty_org, on_qty_remain, value_thresholds, curr_dataset):
    """
    大金额挂单量比例因子
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
        bid_idx = (on_side == 0) & (on_qty_org * on_px / 10000 >= T)
        total_bid_volume = np.sum(on_qty_remain[on_side == 0])
        if total_bid_volume > 0:
            curr_dataset[index, 0] = np.sum(on_qty_remain[bid_idx]) / total_bid_volume
        else:
            curr_dataset[index, 0] = 0

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_qty_org * on_px / 10000 >= T)
        total_ask_volume = np.sum(on_qty_remain[on_side == 1])
        if total_ask_volume > 0:
            curr_dataset[index, 1] = np.sum(on_qty_remain[ask_idx]) / total_ask_volume
        else:
            curr_dataset[index, 1] = 0

        index += 1

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def WeightedCenterPriceDeviation(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    加权重心偏移因子
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2

    # Bid侧计算
    bid_idx = (on_side == 0)
    bid_weighted_price = np.sum(on_px[bid_idx] * (on_qty_remain[bid_idx] * on_px[bid_idx] / 10000)) / np.sum(
        on_qty_remain[bid_idx] * on_px[bid_idx] / 10000
    ) if np.any(bid_idx) else np.nan
    curr_dataset[0, 0] = bid_weighted_price - mid_price

    # Ask侧计算
    ask_idx = (on_side == 1)
    ask_weighted_price = np.sum(on_px[ask_idx] * (on_qty_remain[ask_idx] * on_px[ask_idx] / 10000)) / np.sum(
        on_qty_remain[ask_idx] * on_px[ask_idx] / 10000
    ) if np.any(ask_idx) else np.nan
    curr_dataset[0, 1] = ask_weighted_price - mid_price
    
# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def WeightedCenterPriceVariance(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    加权重心方差因子
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    bid_idx = (on_side == 0)
    if np.any(bid_idx):
        weights = on_qty_remain[bid_idx] * on_px[bid_idx] / 10000
        bid_weighted_price = np.sum(on_px[bid_idx] * weights) / np.sum(weights)
        bid_variance = np.sum(weights * (on_px[bid_idx] - bid_weighted_price) ** 2) / np.sum(weights)
        curr_dataset[0, 0] = bid_variance
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧计算
    ask_idx = (on_side == 1)
    if np.any(ask_idx):
        weights = on_qty_remain[ask_idx] * on_px[ask_idx] / 10000
        ask_weighted_price = np.sum(on_px[ask_idx] * weights) / np.sum(weights)
        ask_variance = np.sum(weights * (on_px[ask_idx] - ask_weighted_price) ** 2) / np.sum(weights)
        curr_dataset[0, 1] = ask_variance
    else:
        curr_dataset[0, 1] = np.nan

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # skewness_weights
    types.float64[:, :]  # curr_dataset
))
def SkewnessWeightedCenterPriceDeviation(best_px, on_side, on_px, on_qty_remain, skewness_weights, curr_dataset):
    """
    偏度加权重心价格及偏移因子
    - skewness_weights: 用于调整偏度的加权参数
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2

    index = 0
    for alpha in skewness_weights:
        # Bid侧计算
        bid_idx = (on_side == 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx] * on_px[bid_idx] / 10000
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                bid_weighted_price = np.sum(on_px[bid_idx] * weights) / weight_sum
                variance = np.sum(weights * (on_px[bid_idx] - bid_weighted_price) ** 2) / weight_sum
                if variance > 0:
                    skewness = np.sum(weights * (on_px[bid_idx] - bid_weighted_price) ** 3) / (variance ** 1.5 * weight_sum)
                    bid_skewed_price = bid_weighted_price + alpha * skewness
                    curr_dataset[index, 0] = bid_skewed_price - mid_price
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx] * on_px[ask_idx] / 10000
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                ask_weighted_price = np.sum(on_px[ask_idx] * weights) / weight_sum
                variance = np.sum(weights * (on_px[ask_idx] - ask_weighted_price) ** 2) / weight_sum
                if variance > 0:
                    skewness = np.sum(weights * (on_px[ask_idx] - ask_weighted_price) ** 3) / (variance ** 1.5 * weight_sum)
                    ask_skewed_price = ask_weighted_price + alpha * skewness
                    curr_dataset[index, 1] = ask_skewed_price - mid_price
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

# @timeit        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # quantiles
    types.float64[:, :]  # curr_dataset
))
def WeightedQuantilePriceDeviation(best_px, on_side, on_px, on_qty_remain, quantiles, curr_dataset):
    """
    加权分位数价格及偏移因子
    - quantiles: 分位数列表 (如 [0.25, 0.5, 0.75])
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2

    # 初始化索引
    index = 0

    for q in quantiles:
        # Bid侧计算
        bid_idx = (on_side == 0)
        if np.any(bid_idx):
            weights = on_qty_remain[bid_idx] * on_px[bid_idx] / 10000
            cumulative_weight = np.cumsum(weights[np.argsort(on_px[bid_idx])])
            total_weight = np.sum(weights)
            sorted_px = on_px[bid_idx][np.argsort(on_px[bid_idx])]
            bid_quantile_price = sorted_px[np.searchsorted(cumulative_weight / total_weight, q, side="right")]
            curr_dataset[index, 0] = bid_quantile_price - mid_price
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1)
        if np.any(ask_idx):
            weights = on_qty_remain[ask_idx] * on_px[ask_idx] / 10000
            cumulative_weight = np.cumsum(weights[np.argsort(on_px[ask_idx])])
            total_weight = np.sum(weights)
            sorted_px = on_px[ask_idx][np.argsort(on_px[ask_idx])]
            ask_quantile_price = sorted_px[np.searchsorted(cumulative_weight / total_weight, q, side="right")]
            curr_dataset[index, 1] = ask_quantile_price - mid_price
        else:
            curr_dataset[index, 1] = np.nan

        index += 1

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:, :]  # curr_dataset
))
def OrderAmountVarianceRatio(best_px, on_side, on_px, on_qty_remain, price_range, curr_dataset):
    """
    计算挂单金额的均值与方差比值，以刻画价格区间内分布的离散性。
    - price_range: 用于限制挂单价格在一定范围内。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历价格区间参数
    for pr in price_range:
        # Bid侧计算
        bid_idx = (on_side == 0) & (on_px >= bid1 - pr) & (on_px <= bid1 + pr)
        bid_amounts = on_qty_remain[bid_idx] * on_px[bid_idx] / 10000
        if bid_amounts.size > 0:
            mean_bid = np.mean(bid_amounts)
            var_bid = np.var(bid_amounts)
            curr_dataset[index, 0] = var_bid / mean_bid if mean_bid != 0 else np.nan
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_idx = (on_side == 1) & (on_px >= ask1 - pr) & (on_px <= ask1 + pr)
        ask_amounts = on_qty_remain[ask_idx] * on_px[ask_idx] / 10000
        if ask_amounts.size > 0:
            mean_ask = np.mean(ask_amounts)
            var_ask = np.var(ask_amounts)
            curr_dataset[index, 1] = var_ask / mean_ask if mean_ask != 0 else np.nan
        else:
            curr_dataset[index, 1] = np.nan

        # 更新行索引
        index += 1

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderAmountCVR(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额均值方差比因子计算
    - 计算挂单金额的均值方差比，反映挂单分布的离散程度。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：买1或卖1价格无效时填充NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # Bid侧计算
    bid_idx = (on_side == 0)
    bid_amount = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000  # 挂单金额
    if bid_amount.size > 0:
        mean_bid = np.mean(bid_amount)
        var_bid = np.var(bid_amount)
        curr_dataset[index, 0] = var_bid / mean_bid if mean_bid > 0 else np.nan
    else:
        curr_dataset[index, 0] = np.nan

    # Ask侧计算
    ask_idx = (on_side == 1)
    ask_amount = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000  # 挂单金额
    if ask_amount.size > 0:
        mean_ask = np.mean(ask_amount)
        var_ask = np.var(ask_amount)
        curr_dataset[index, 1] = var_ask / mean_ask if mean_ask > 0 else np.nan
    else:
        curr_dataset[index, 1] = np.nan

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderAmountSkewness(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额偏态系数因子计算
    - 计算挂单金额的偏态系数，衡量分布的对称性。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：买1或卖1价格无效时填充NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    bid_idx = (on_side == 0)
    bid_amount = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000  # 挂单金额
    if bid_amount.size > 1:  # 至少需要两个数据点
        mean_bid = np.mean(bid_amount)
        std_bid = np.std(bid_amount)
        if std_bid > 0:
            skewness_bid = np.mean(((bid_amount - mean_bid) / std_bid) ** 3)
            curr_dataset[0, 0] = skewness_bid
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧计算
    ask_idx = (on_side == 1)
    ask_amount = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000  # 挂单金额
    if ask_amount.size > 1:  # 至少需要两个数据点
        mean_ask = np.mean(ask_amount)
        std_ask = np.std(ask_amount)
        if std_ask > 0:
            skewness_ask = np.mean(((ask_amount - mean_ask) / std_ask) ** 3)
            curr_dataset[0, 1] = skewness_ask
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = np.nan
        
# @timeit        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderAmountKurtosis(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额峰度因子计算
    - 计算挂单金额的峰度，衡量分布的尖峰程度。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：买1或卖1价格无效时填充NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    bid_idx = (on_side == 0)
    bid_amount = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000  # 挂单金额
    if bid_amount.size > 1:  # 至少需要两个数据点
        mean_bid = np.mean(bid_amount)
        std_bid = np.std(bid_amount)
        if std_bid > 0:
            kurtosis_bid = np.mean(((bid_amount - mean_bid) / std_bid) ** 4) - 3
            curr_dataset[0, 0] = kurtosis_bid
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = np.nan

    # Ask侧计算
    ask_idx = (on_side == 1)
    ask_amount = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000  # 挂单金额
    if ask_amount.size > 1:  # 至少需要两个数据点
        mean_ask = np.mean(ask_amount)
        std_ask = np.std(ask_amount)
        if std_ask > 0:
            kurtosis_ask = np.mean(((ask_amount - mean_ask) / std_ask) ** 4) - 3
            curr_dataset[0, 1] = kurtosis_ask
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = np.nan

# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # lower_percentiles (list of lower percentiles, e.g., [0.1, 0.25])
    types.float64[:],  # upper_percentiles (list of upper percentiles, e.g., [0.75, 0.9])
    types.float64[:, :]  # curr_dataset
))
def OrderAmountIQR(best_px, on_side, on_px, on_qty_remain, lower_percentiles, upper_percentiles, curr_dataset):
    """
    挂单金额分位数间距因子计算
    - 计算挂单金额的特定分位数间距 (如90%-10%, 75%-25%)
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：买1或卖1价格无效时填充NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历分位数组合
    for p1 in lower_percentiles:
        for p2 in upper_percentiles:
            if p1 >= p2:
                continue  # 保证分位数的逻辑顺序
            
            # Bid侧计算
            bid_idx = (on_side == 0)
            bid_amount = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000  # 挂单金额
            if bid_amount.size > 0:
                q1_bid = np.percentile(bid_amount, p1 * 100)
                q2_bid = np.percentile(bid_amount, p2 * 100)
                curr_dataset[index, 0] = q2_bid - q1_bid
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1)
            ask_amount = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000  # 挂单金额
            if ask_amount.size > 0:
                q1_ask = np.percentile(ask_amount, p1 * 100)
                q2_ask = np.percentile(ask_amount, p2 * 100)
                curr_dataset[index, 1] = q2_ask - q1_ask
            else:
                curr_dataset[index, 1] = np.nan

            # 更新行索引
            index += 1


# @timeit
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # percent_changes (list of percentage changes, e.g., [-0.02, -0.01, 0.01, 0.02])
    types.float64[:, :]  # curr_dataset
))
def OrderAmountPercentSensitivity(best_px, on_side, on_px, on_qty_remain, percent_changes, curr_dataset):
    """
    挂单金额价格百分比敏感性因子计算
    - 计算挂单金额对价格百分比变化的敏感性 (Δ金额 / Δ价格百分比)。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：买1或卖1价格无效时填充NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历百分比变化
    for percent_change in percent_changes:
        # Bid侧计算
        bid_new_price = int(bid1 * (1 + percent_change))
        bid_idx = (on_side == 0) & (on_px <= bid_new_price)
        original_bid_amount = (on_px[on_side == 0] * on_qty_remain[on_side == 0]) / 10000
        adjusted_bid_amount = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000

        if original_bid_amount.size > 0 and adjusted_bid_amount.size > 0:
            amount_change_bid = np.sum(adjusted_bid_amount) - np.sum(original_bid_amount)
            if percent_change != 0:
                curr_dataset[index, 0] = amount_change_bid / percent_change
            else:
                curr_dataset[index, 0] = np.nan
        else:
            curr_dataset[index, 0] = np.nan

        # Ask侧计算
        ask_new_price = int(ask1 * (1 + percent_change))
        ask_idx = (on_side == 1) & (on_px >= ask_new_price)
        original_ask_amount = (on_px[on_side == 1] * on_qty_remain[on_side == 1]) / 10000
        adjusted_ask_amount = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000

        if original_ask_amount.size > 0 and adjusted_ask_amount.size > 0:
            amount_change_ask = np.sum(adjusted_ask_amount) - np.sum(original_ask_amount)
            if percent_change != 0:
                curr_dataset[index, 1] = amount_change_ask / percent_change
            else:
                curr_dataset[index, 1] = np.nan
        else:
            curr_dataset[index, 1] = np.nan

        # 更新行索引
        index += 1

