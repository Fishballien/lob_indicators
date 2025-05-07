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
    types.float64[:],  # long_order_amount_threshold
    types.float32[:, :]  # curr_dataset
))
def LongOrderAmountRatio(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, long_order_amount_threshold, curr_dataset):
    """
    计算漫长订单成交量占比。遍历不同的成交量阈值计算成交量占比。
    - best_px: 当前买1、卖1价格
    - on_side: 订单方向，0为买单，1为卖单
    - on_px: 订单价格
    - on_qty_org: 订单原始数量
    - on_qty_remain: 订单剩余数量
    - on_ts_org: 订单挂单时间戳
    - ts: 当前时间戳
    - long_order_amount_threshold: 漫长订单金额阈值
    - curr_dataset: 存储计算结果的数组，形状为n*2，0列存储Bid侧结果，1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 遍历所有的阈值
    index = 0
    for threshold in long_order_amount_threshold:
        # 计算时间差，判断漫长订单
        residue_time = ts - on_ts_org
        is_long_order = residue_time >= threshold  # 判断是否为漫长订单

        # 筛选漫长订单
        long_order_qty_remain = on_qty_remain[is_long_order]
        long_order_amount_remain = (on_px[is_long_order] * long_order_qty_remain) / 10000  # 计算挂单金额

        if long_order_qty_remain.size == 0:  # 如果没有漫长订单，填充 NaN
            curr_dataset[index, 0] = np.nan
            curr_dataset[index, 1] = np.nan
        else:
            # 计算漫长订单成交量占比
            total_amount = np.sum(on_px * on_qty_remain) / 10000
            long_order_amount = np.sum(long_order_amount_remain)
            long_order_amount_ratio = long_order_amount / total_amount if total_amount > 0 else np.nan
            curr_dataset[index, 0] = long_order_amount_ratio
            curr_dataset[index, 1] = long_order_amount_ratio

        index += 1
        
        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,  # ts
    types.float64[:],  # long_order_time_threshold
    types.float32[:, :]  # curr_dataset
))
def LongOrderPriceDeviation(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, long_order_time_threshold, curr_dataset):
    """
    计算漫长订单的挂单金额加权价与中间价的偏离的绝对值。
    - best_px: 当前买1、卖1价格
    - on_side: 订单方向，0为买单，1为卖单
    - on_px: 订单价格
    - on_qty_org: 订单原始数量
    - on_qty_remain: 订单剩余数量
    - on_ts_org: 订单挂单时间戳
    - ts: 当前时间戳
    - long_order_time_threshold: 漫长订单时间阈值
    - curr_dataset: 存储计算结果的数组，形状为n*2，0列存储Bid侧结果，1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 中间价
    mid_price = (bid1 + ask1) / 2

    # 遍历所有的时间阈值
    index = 0
    for threshold in long_order_time_threshold:
        # 计算时间差，判断漫长订单
        residue_time = ts - on_ts_org
        is_long_order = residue_time >= threshold  # 判断是否为漫长订单

        # 筛选漫长买单（买单的挂单价格 <= 买一价格）
        long_buy_idx = (on_side == 0) & is_long_order
        long_buy_qty_remain = on_qty_remain[long_buy_idx]
        long_buy_px = on_px[long_buy_idx]

        # 筛选漫长卖单（卖单的挂单价格 >= 卖一价格）
        long_sell_idx = (on_side == 1) & is_long_order
        long_sell_qty_remain = on_qty_remain[long_sell_idx]
        long_sell_px = on_px[long_sell_idx]

        # 计算加权价格
        weighted_price_buy = np.sum(long_buy_px * long_buy_qty_remain) / np.sum(long_buy_qty_remain) if np.any(long_buy_qty_remain) else np.nan
        weighted_price_sell = np.sum(long_sell_px * long_sell_qty_remain) / np.sum(long_sell_qty_remain) if np.any(long_sell_qty_remain) else np.nan

        # 计算价格偏离的绝对值
        if not np.isnan(weighted_price_buy):
            buy_price_deviation = np.abs(weighted_price_buy - mid_price) / mid_price
        else:
            buy_price_deviation = np.nan
        
        if not np.isnan(weighted_price_sell):
            sell_price_deviation = np.abs(weighted_price_sell - mid_price) / mid_price
        else:
            sell_price_deviation = np.nan

        # 更新结果
        curr_dataset[index, 0] = buy_price_deviation
        curr_dataset[index, 1] = sell_price_deviation

        index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,  # ts
    types.float64[:],  # long_order_time_threshold
    types.float32[:, :]  # curr_dataset
))
def LongOrderConcentrationGini(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, long_order_time_threshold, curr_dataset):
    """
    计算漫长订单的集中度，使用Gini指数衡量。
    - best_px: 当前买1、卖1价格
    - on_side: 订单方向，0为买单，1为卖单
    - on_px: 订单价格
    - on_qty_org: 订单原始数量
    - on_qty_remain: 订单剩余数量
    - on_ts_org: 订单挂单时间戳
    - ts: 当前时间戳
    - long_order_time_threshold: 漫长订单时间阈值
    - curr_dataset: 存储计算结果的数组，形状为n*2，0列存储Bid侧结果，1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 遍历所有的时间阈值
    index = 0
    for threshold in long_order_time_threshold:
        # 计算时间差，判断漫长订单
        residue_time = ts - on_ts_org
        is_long_order = residue_time >= threshold  # 判断是否为漫长订单

        # 筛选漫长订单的剩余数量
        long_order_qty_remain = on_qty_remain[is_long_order]

        # 如果没有漫长订单，填充 NaN
        if long_order_qty_remain.size == 0:
            curr_dataset[index, 0] = np.nan
            curr_dataset[index, 1] = np.nan
        else:
            # 计算基尼系数
            long_order_qty_remain_sorted = np.sort(long_order_qty_remain)
            n = len(long_order_qty_remain_sorted)
            gini_index = 1 - 2 * np.sum(long_order_qty_remain_sorted * (n - np.arange(1, n + 1))) / (n * np.sum(long_order_qty_remain_sorted))

            # 更新结果
            curr_dataset[index, 0] = gini_index
            curr_dataset[index, 1] = gini_index

        index += 1
