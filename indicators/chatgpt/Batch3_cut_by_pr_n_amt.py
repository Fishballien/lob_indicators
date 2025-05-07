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
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def WeightedPriceCenterDistance(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：计算金额加权重心价格到中间价的距离。
    参数：
        - price_range：价格范围百分比切片
        - amount_threshold：初始挂单金额阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：无效价格
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2.0
    index = 0

    # 遍历价格范围和金额阈值
    for p_range in price_range:
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)
        
        for a_threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound) & (
                on_px * on_qty_org / 10000 > a_threshold
            )
            if np.any(bid_idx):  # 筛选结果非空
                bid_weights = on_qty_remain[bid_idx] * on_px[bid_idx]  # 挂单金额权重（剩余量）
                bid_prices = on_px[bid_idx]
                weighted_price = np.sum(bid_weights * bid_prices) / np.sum(bid_weights)
                curr_dataset[index, 0] = np.abs(weighted_price - mid_price)
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound) & (
                on_px * on_qty_org / 10000 > a_threshold
            )
            if np.any(ask_idx):  # 筛选结果非空
                ask_weights = on_qty_remain[ask_idx] * on_px[ask_idx]  # 挂单金额权重（剩余量）
                ask_prices = on_px[ask_idx]
                weighted_price = np.sum(ask_weights * ask_prices) / np.sum(ask_weights)
                curr_dataset[index, 1] = np.abs(weighted_price - mid_price)
            else:
                curr_dataset[index, 1] = np.nan

            # 更新索引
            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def LogScaledPriceCenterDistance(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：计算对数尺度加权重心价格到中间价的距离。
    参数：
        - price_range：价格范围百分比切片
        - amount_threshold：初始挂单金额阈值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：无效价格
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2.0
    index = 0

    # 遍历价格范围和金额阈值
    for p_range in price_range:
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for a_threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound) & (
                on_px * on_qty_org / 10000 > a_threshold
            )
            if np.any(bid_idx):  # 筛选结果非空
                bid_weights = on_qty_remain[bid_idx] * on_px[bid_idx]  # 剩余挂单金额权重
                bid_log_prices = np.log(on_px[bid_idx])  # 对数价格
                log_scaled_price = np.sum(bid_weights * bid_log_prices) / np.sum(bid_weights)
                curr_dataset[index, 0] = np.abs(log_scaled_price - np.log(mid_price))
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound) & (
                on_px * on_qty_org / 10000 > a_threshold
            )
            if np.any(ask_idx):  # 筛选结果非空
                ask_weights = on_qty_remain[ask_idx] * on_px[ask_idx]  # 剩余挂单金额权重
                ask_log_prices = np.log(on_px[ask_idx])  # 对数价格
                log_scaled_price = np.sum(ask_weights * ask_log_prices) / np.sum(ask_weights)
                curr_dataset[index, 1] = np.abs(log_scaled_price - np.log(mid_price))
            else:
                curr_dataset[index, 1] = np.nan

            # 更新索引
            index += 1
            
            
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:],  # quantiles
    types.float64[:, :]  # curr_dataset
))
def PriceQuantileDeviation(best_px, on_side, on_px, on_qty_org, price_range, amount_threshold, quantiles, curr_dataset):
    """
    因子：挂单价格分位数偏离（带价格范围与金额限制）
    - price_range: 中间价上下浮动范围的百分比
    - amount_threshold: 初始挂单金额限制（单位：元）
    - quantiles: 分位数列表
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2  # 中间价
    index = 0

    for p_range in price_range:
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for threshold in amount_threshold:
            for q in quantiles:
                # Bid侧计算
                bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound)
                if np.any(bid_idx):
                    bid_px = on_px[bid_idx]
                    bid_qty = on_qty_org[bid_idx]
                    bid_amt = bid_px * bid_qty / 10000  # 挂单金额计算
                    valid_idx = bid_amt > threshold
                    if np.any(valid_idx):
                        curr_dataset[index, 0] = np.percentile(bid_px[valid_idx], q * 100) - mid_price
                    else:
                        curr_dataset[index, 0] = np.nan
                else:
                    curr_dataset[index, 0] = np.nan

                # Ask侧计算
                ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound)
                if np.any(ask_idx):
                    ask_px = on_px[ask_idx]
                    ask_qty = on_qty_org[ask_idx]
                    ask_amt = ask_px * ask_qty / 10000  # 挂单金额计算
                    valid_idx = ask_amt > threshold
                    if np.any(valid_idx):
                        curr_dataset[index, 1] = np.percentile(ask_px[valid_idx], q * 100) - mid_price
                    else:
                        curr_dataset[index, 1] = np.nan
                else:
                    curr_dataset[index, 1] = np.nan

                index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def TotalAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    挂单总金额因子
    参数：
        - price_range：价格范围，按中间价的百分比切片
        - amount_threshold：初始挂单金额的阈值
    """
    mid_price = (best_px[0] + best_px[1]) / 2

    if mid_price == 0:  # 无效价格处理
        curr_dataset[:, :] = np.nan
        return

    index = 0

    for pr in price_range:  # 遍历价格范围
        for thres in amount_threshold:  # 遍历金额阈值
            # 计算价格区间
            price_lower = mid_price * (1 - pr)
            price_upper = mid_price * (1 + pr)

            # Bid侧
            bid_idx = (on_side == 0) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(bid_idx):  # 检查筛选结果
                bid_initial_amount = on_qty_org[bid_idx] * on_px[bid_idx] / 10000  # 使用初始挂单量计算金额
                bid_valid_idx = bid_initial_amount > thres  # 应用金额阈值
                if np.any(bid_valid_idx):
                    bid_remain_amount = on_qty_remain[bid_idx][bid_valid_idx] * on_px[bid_idx][bid_valid_idx] / 10000
                    curr_dataset[index, 0] = np.sum(bid_remain_amount)  # 使用剩余挂单量计算总金额
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧
            ask_idx = (on_side == 1) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(ask_idx):  # 检查筛选结果
                ask_initial_amount = on_qty_org[ask_idx] * on_px[ask_idx] / 10000  # 使用初始挂单量计算金额
                ask_valid_idx = ask_initial_amount > thres  # 应用金额阈值
                if np.any(ask_valid_idx):
                    ask_remain_amount = on_qty_remain[ask_idx][ask_valid_idx] * on_px[ask_idx][ask_valid_idx] / 10000
                    curr_dataset[index, 1] = np.sum(ask_remain_amount)  # 使用剩余挂单量计算总金额
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1

        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def PriceDeviationAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    价格分布偏离加权金额因子
    参数：
        - price_range：价格范围，按中间价的百分比切片
        - amount_threshold：初始挂单金额的阈值
    """
    mid_price = (best_px[0] + best_px[1]) / 2

    if mid_price == 0:  # 无效价格处理
        curr_dataset[:, :] = np.nan
        return

    index = 0

    for pr in price_range:  # 遍历价格范围
        for thres in amount_threshold:  # 遍历金额阈值
            # 计算价格区间
            price_lower = mid_price * (1 - pr)
            price_upper = mid_price * (1 + pr)

            # Bid侧
            bid_idx = (on_side == 0) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(bid_idx):  # 确保筛选结果非空
                bid_initial_amount = on_qty_org[bid_idx] * on_px[bid_idx] / 10000  # 初始挂单金额
                bid_valid_idx = bid_initial_amount > thres  # 应用金额阈值
                if np.any(bid_valid_idx):
                    bid_remain_amount = on_qty_remain[bid_idx][bid_valid_idx] * on_px[bid_idx][bid_valid_idx] / 10000
                    price_deviation = np.abs(on_px[bid_idx][bid_valid_idx] - mid_price)
                    curr_dataset[index, 0] = np.sum(price_deviation * bid_remain_amount)
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧
            ask_idx = (on_side == 1) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(ask_idx):  # 确保筛选结果非空
                ask_initial_amount = on_qty_org[ask_idx] * on_px[ask_idx] / 10000  # 初始挂单金额
                ask_valid_idx = ask_initial_amount > thres  # 应用金额阈值
                if np.any(ask_valid_idx):
                    ask_remain_amount = on_qty_remain[ask_idx][ask_valid_idx] * on_px[ask_idx][ask_valid_idx] / 10000
                    price_deviation = np.abs(on_px[ask_idx][ask_valid_idx] - mid_price)
                    curr_dataset[index, 1] = np.sum(price_deviation * ask_remain_amount)
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def WeightedDispersionAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    加权局部金额离散度因子
    参数：
        - price_range：价格范围，按中间价的百分比切片
        - amount_threshold：初始挂单金额的阈值
    """
    mid_price = (best_px[0] + best_px[1]) / 2

    if mid_price == 0:  # 无效价格处理
        curr_dataset[:, :] = np.nan
        return

    index = 0

    for pr in price_range:  # 遍历价格范围
        for thres in amount_threshold:  # 遍历金额阈值
            # 计算价格区间
            price_lower = mid_price * (1 - pr)
            price_upper = mid_price * (1 + pr)

            # Bid侧
            bid_idx = (on_side == 0) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(bid_idx):  # 确保筛选结果非空
                bid_initial_amount = on_qty_org[bid_idx] * on_px[bid_idx] / 10000  # 初始挂单金额
                bid_valid_idx = bid_initial_amount > thres  # 应用金额阈值
                if np.any(bid_valid_idx):
                    bid_prices = on_px[bid_idx][bid_valid_idx]
                    bid_weights = on_qty_remain[bid_idx][bid_valid_idx]  # 使用剩余挂单量作为权重
                    local_mean = np.sum(bid_prices * bid_weights) / np.sum(bid_weights)
                    curr_dataset[index, 0] = np.sum(bid_weights * (bid_prices - local_mean) ** 2) / np.sum(bid_weights)
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧
            ask_idx = (on_side == 1) & (on_px >= price_lower) & (on_px <= price_upper)
            if np.any(ask_idx):  # 确保筛选结果非空
                ask_initial_amount = on_qty_org[ask_idx] * on_px[ask_idx] / 10000  # 初始挂单金额
                ask_valid_idx = ask_initial_amount > thres  # 应用金额阈值
                if np.any(ask_valid_idx):
                    ask_prices = on_px[ask_idx][ask_valid_idx]
                    ask_weights = on_qty_remain[ask_idx][ask_valid_idx]  # 使用剩余挂单量作为权重
                    local_mean = np.sum(ask_prices * ask_weights) / np.sum(ask_weights)
                    curr_dataset[index, 1] = np.sum(ask_weights * (ask_prices - local_mean) ** 2) / np.sum(ask_weights)
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def LargeOrderProportion(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：大额挂单占比
    - price_range: 价格范围百分比，控制挂单价格范围
    - amount_threshold: 初始挂单金额限制（单位：元）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for p_range in price_range:
        mid_price = (bid1 + ask1) / 2
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(bid_idx):
                bid_initial_amount = on_qty_org[bid_idx] * on_px[bid_idx] / 10000  # 初始挂单金额
                valid_bid_idx = bid_initial_amount > threshold  # 应用金额阈值
                if np.any(valid_bid_idx):
                    bid_weights = on_qty_remain[bid_idx][valid_bid_idx]  # 剩余挂单量
                    total_bid_weights = on_qty_remain[bid_idx]  # 总剩余挂单量
                    curr_dataset[index, 0] = (
                        np.sum(bid_weights) / np.sum(total_bid_weights) if np.sum(total_bid_weights) > 0 else np.nan
                    )
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(ask_idx):
                ask_initial_amount = on_qty_org[ask_idx] * on_px[ask_idx] / 10000  # 初始挂单金额
                valid_ask_idx = ask_initial_amount > threshold  # 应用金额阈值
                if np.any(valid_ask_idx):
                    ask_weights = on_qty_remain[ask_idx][valid_ask_idx]  # 剩余挂单量
                    total_ask_weights = on_qty_remain[ask_idx]  # 总剩余挂单量
                    curr_dataset[index, 1] = (
                        np.sum(ask_weights) / np.sum(total_ask_weights) if np.sum(total_ask_weights) > 0 else np.nan
                    )
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1

        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def OrderDepthConcentration(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：挂单深度集中度
    - price_range: 价格范围百分比，控制挂单价格范围
    - amount_threshold: 初始挂单金额限制（单位：元）
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for p_range in price_range:
        mid_price = (bid1 + ask1) / 2
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(bid_idx):
                bid_initial_amount = on_qty_org[bid_idx] * on_px[bid_idx] / 10000  # 初始挂单金额
                valid_bid_idx = bid_initial_amount > threshold  # 应用金额阈值
                if np.any(valid_bid_idx):
                    valid_qty = on_qty_remain[bid_idx][valid_bid_idx]  # 使用剩余挂单量
                    numerator = np.sum(valid_qty**2)
                    denominator = np.sum(valid_qty)**2
                    curr_dataset[index, 0] = numerator / denominator if denominator > 0 else np.nan
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(ask_idx):
                ask_initial_amount = on_qty_org[ask_idx] * on_px[ask_idx] / 10000  # 初始挂单金额
                valid_ask_idx = ask_initial_amount > threshold  # 应用金额阈值
                if np.any(valid_ask_idx):
                    valid_qty = on_qty_remain[ask_idx][valid_ask_idx]  # 使用剩余挂单量
                    numerator = np.sum(valid_qty**2)
                    denominator = np.sum(valid_qty)**2
                    curr_dataset[index, 1] = numerator / denominator if denominator > 0 else np.nan
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def OverallAmountHHI(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：整体金额HHI
    - 使用金额计算HHI（Herfindahl-Hirschman指数），衡量金额的集中性
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2
    index = 0

    for p_range in price_range:
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(bid_idx):
                bid_amt = on_px[bid_idx] * on_qty_org[bid_idx] / 10000  # 初始挂单金额计算
                valid_idx = bid_amt > threshold
                if np.any(valid_idx):
                    valid_amt = on_px[bid_idx][valid_idx] * on_qty_remain[bid_idx][valid_idx] / 10000  # 剩余挂单金额
                    proportions = valid_amt / np.sum(valid_amt)
                    curr_dataset[index, 0] = np.sum(proportions**2)
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(ask_idx):
                ask_amt = on_px[ask_idx] * on_qty_org[ask_idx] / 10000  # 初始挂单金额计算
                valid_idx = ask_amt > threshold
                if np.any(valid_idx):
                    valid_amt = on_px[ask_idx][valid_idx] * on_qty_remain[ask_idx][valid_idx] / 10000  # 剩余挂单金额
                    proportions = valid_amt / np.sum(valid_amt)
                    curr_dataset[index, 1] = np.sum(proportions**2)
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def PriceWeightedAmountHHI(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：价格加权金额HHI
    - 使用金额和价格距离计算加权HHI
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2
    index = 0

    for p_range in price_range:
        lower_bound = mid_price * (1 - p_range)
        upper_bound = mid_price * (1 + p_range)

        for threshold in amount_threshold:
            # Bid侧计算
            bid_idx = (on_side == 0) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(bid_idx):
                bid_amt = on_px[bid_idx] * on_qty_org[bid_idx] / 10000  # 初始挂单金额计算
                bid_dist = np.abs(on_px[bid_idx] - mid_price)
                valid_idx = bid_amt > threshold
                if np.any(valid_idx):
                    valid_amt = on_px[bid_idx][valid_idx] * on_qty_remain[bid_idx][valid_idx] / 10000  # 剩余挂单金额
                    valid_dist = bid_dist[valid_idx]
                    weights = valid_amt * valid_dist
                    proportions = weights / np.sum(weights)
                    curr_dataset[index, 0] = np.sum(proportions**2)
                else:
                    curr_dataset[index, 0] = np.nan
            else:
                curr_dataset[index, 0] = np.nan

            # Ask侧计算
            ask_idx = (on_side == 1) & (on_px >= lower_bound) & (on_px <= upper_bound)
            if np.any(ask_idx):
                ask_amt = on_px[ask_idx] * on_qty_org[ask_idx] / 10000  # 初始挂单金额计算
                ask_dist = np.abs(on_px[ask_idx] - mid_price)
                valid_idx = ask_amt > threshold
                if np.any(valid_idx):
                    valid_amt = on_px[ask_idx][valid_idx] * on_qty_remain[ask_idx][valid_idx] / 10000  # 剩余挂单金额
                    valid_dist = ask_dist[valid_idx]
                    weights = valid_amt * valid_dist
                    proportions = weights / np.sum(weights)
                    curr_dataset[index, 1] = np.sum(proportions**2)
                else:
                    curr_dataset[index, 1] = np.nan
            else:
                curr_dataset[index, 1] = np.nan

            index += 1

            
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def MeanVarianceRatio(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：均值-方差比（MVR）。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2
    index = 0

    for pr in price_range:
        for at in amount_threshold:
            lower_bound = mid_price * (1 - pr)
            upper_bound = mid_price * (1 + pr)

            for side in [0, 1]:  # 分别计算 Bid 和 Ask
                mask = (on_side == side) & (on_px >= lower_bound) & (on_px <= upper_bound)
                if np.any(mask):
                    initial_amount = (on_qty_org[mask] * on_px[mask]) / 10000
                    valid_idx = initial_amount >= at
                    if np.any(valid_idx):
                        valid_qty = on_qty_remain[mask][valid_idx]  # 使用剩余挂单量
                        mean_val = valid_qty.mean()
                        var_val = valid_qty.var()
                        curr_dataset[index, side] = var_val / mean_val if mean_val > 0 else np.nan
                    else:
                        curr_dataset[index, side] = np.nan
                else:
                    curr_dataset[index, side] = np.nan
            index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.float64[:],  # price_range
    types.float64[:],  # amount_threshold
    types.float64[:, :]  # curr_dataset
))
def ConcentrationDispersionRatio(best_px, on_side, on_px, on_qty_org, on_qty_remain, price_range, amount_threshold, curr_dataset):
    """
    因子：集中-离散分布比率（CDR）。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2
    index = 0

    for pr in price_range:
        for at in amount_threshold:
            core_lower = mid_price * (1 - pr)
            core_upper = mid_price * (1 + pr)

            for side in [0, 1]:  # 分别计算 Bid 和 Ask
                core_mask = (on_side == side) & (on_px >= core_lower) & (on_px <= core_upper)
                non_core_mask = (on_side == side) & ~core_mask

                core_initial_amount = (on_qty_org[core_mask] * on_px[core_mask]) / 10000
                non_core_initial_amount = (on_qty_org[non_core_mask] * on_px[non_core_mask]) / 10000

                core_valid_idx = core_initial_amount >= at
                non_core_valid_idx = non_core_initial_amount >= at

                if np.any(core_valid_idx) and np.any(non_core_valid_idx):
                    core_qty = on_qty_remain[core_mask][core_valid_idx]  # 核心区域剩余挂单量
                    non_core_qty = on_qty_remain[non_core_mask][non_core_valid_idx]  # 非核心区域剩余挂单量
                    core_var = core_qty.var()
                    non_core_var = non_core_qty.var()
                    curr_dataset[index, side] = core_var / non_core_var if non_core_var > 0 else np.nan
                else:
                    curr_dataset[index, side] = np.nan
            index += 1



