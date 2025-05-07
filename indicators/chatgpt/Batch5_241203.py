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
    types.float64[:, :]  # curr_dataset
))
def OrderAmountGini(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额基尼系数计算因子：衡量挂单金额分布的集中度。

    Parameters:
    - best_px: 当前买1（Bid）和卖1（Ask）价格
    - on_side: 当前挂单的方向 (0=Bid, 1=Ask)
    - on_px: 当前挂单价格
    - on_qty_remain: 当前剩余挂单数量
    - curr_dataset: 用于存储结果的数组，形状为 n*2，第0列为Bid侧，第1列为Ask侧
    """

    # 边界处理：买1或卖1价格无效
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化结果索引
    index = 0

    # Bid侧计算
    bid_idx = on_side == 0
    bid_amounts = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000
    if bid_amounts.size > 0:
        total_amount_bid = np.sum(bid_amounts)
        sorted_bid_amounts = np.sort(bid_amounts)
        n_bid = len(sorted_bid_amounts)
        gini_bid = 0
        for i in range(n_bid):
            for j in range(n_bid):
                gini_bid += abs(sorted_bid_amounts[i] - sorted_bid_amounts[j])
        gini_bid /= (2 * n_bid * total_amount_bid) if total_amount_bid > 0 else np.nan
        curr_dataset[index, 0] = gini_bid
    else:
        curr_dataset[index, 0] = np.nan

    # Ask侧计算
    ask_idx = on_side == 1
    ask_amounts = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000
    if ask_amounts.size > 0:
        total_amount_ask = np.sum(ask_amounts)
        sorted_ask_amounts = np.sort(ask_amounts)
        n_ask = len(sorted_ask_amounts)
        gini_ask = 0
        for i in range(n_ask):
            for j in range(n_ask):
                gini_ask += abs(sorted_ask_amounts[i] - sorted_ask_amounts[j])
        gini_ask /= (2 * n_ask * total_amount_ask) if total_amount_ask > 0 else np.nan
        curr_dataset[index, 1] = gini_ask
    else:
        curr_dataset[index, 1] = np.nan
        
        
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderAmountHHI(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额赫芬达尔-赫希曼指数（HHI）因子计算。
    
    Parameters:
    - best_px: 当前买1（Bid）和卖1（Ask）价格
    - on_side: 当前挂单的方向 (0=Bid, 1=Ask)
    - on_px: 当前挂单价格
    - on_qty_remain: 当前剩余挂单数量
    - curr_dataset: 用于存储结果的数组，形状为 n*2，第0列为Bid侧，第1列为Ask侧
    """

    # 边界处理：买1或卖1价格无效
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化结果索引
    index = 0

    # Bid侧计算
    bid_idx = on_side == 0
    bid_amounts = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000
    if bid_amounts.size > 0:
        total_amount_bid = np.sum(bid_amounts)
        if total_amount_bid > 0:
            hhi_bid = np.sum((bid_amounts / total_amount_bid) ** 2)
            curr_dataset[index, 0] = hhi_bid
        else:
            curr_dataset[index, 0] = np.nan
    else:
        curr_dataset[index, 0] = np.nan

    # Ask侧计算
    ask_idx = on_side == 1
    ask_amounts = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000
    if ask_amounts.size > 0:
        total_amount_ask = np.sum(ask_amounts)
        if total_amount_ask > 0:
            hhi_ask = np.sum((ask_amounts / total_amount_ask) ** 2)
            curr_dataset[index, 1] = hhi_ask
        else:
            curr_dataset[index, 1] = np.nan
    else:
        curr_dataset[index, 1] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # quantiles
    types.float64[:, :]  # curr_dataset
))
def OrderAmountQuantileConcentration(best_px, on_side, on_px, on_qty_remain, quantiles, curr_dataset):
    """
    挂单金额分位点集中度指标计算因子。
    
    Parameters:
    - best_px: 当前买1（Bid）和卖1（Ask）价格
    - on_side: 当前挂单的方向 (0=Bid, 1=Ask)
    - on_px: 当前挂单价格
    - on_qty_remain: 当前剩余挂单数量
    - quantiles: 需要计算的分位点列表 (如 0.2, 0.5 表示前 20%、50%)
    - curr_dataset: 用于存储结果的数组，形状为 n*2，第0列为Bid侧，第1列为Ask侧
    """

    # 边界处理：买1或卖1价格无效
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧计算
    index = 0
    bid_idx = on_side == 0
    bid_amounts = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000
    if bid_amounts.size > 0:
        total_amount_bid = np.sum(bid_amounts)
        if total_amount_bid > 0:
            sorted_bid_amounts = np.sort(bid_amounts)[::-1]  # 从大到小排序
            cumulative_sum_bid = np.cumsum(sorted_bid_amounts)  # 累计金额
            for q in quantiles:
                threshold_idx = int(np.ceil(len(cumulative_sum_bid) * q)) - 1
                if threshold_idx >= 0:
                    curr_dataset[index, 0] = cumulative_sum_bid[threshold_idx] / total_amount_bid
                else:
                    curr_dataset[index, 0] = np.nan
                index += 1
        else:
            curr_dataset[index:index+len(quantiles), 0] = np.nan
            index += len(quantiles)
    else:
        curr_dataset[index:index+len(quantiles), 0] = np.nan
        index += len(quantiles)

    # Ask侧计算
    index = 0  # 重置 index
    ask_idx = on_side == 1
    ask_amounts = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000
    if ask_amounts.size > 0:
        total_amount_ask = np.sum(ask_amounts)
        if total_amount_ask > 0:
            sorted_ask_amounts = np.sort(ask_amounts)[::-1]  # 从大到小排序
            cumulative_sum_ask = np.cumsum(sorted_ask_amounts)  # 累计金额
            for q in quantiles:
                threshold_idx = int(np.ceil(len(cumulative_sum_ask) * q)) - 1
                if threshold_idx >= 0:
                    curr_dataset[index, 1] = cumulative_sum_ask[threshold_idx] / total_amount_ask
                else:
                    curr_dataset[index, 1] = np.nan
                index += 1
        else:
            curr_dataset[index:index+len(quantiles), 1] = np.nan
            index += len(quantiles)
    else:
        curr_dataset[index:index+len(quantiles), 1] = np.nan
        index += len(quantiles)


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def OrderAmountEntropy(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单金额分布熵因子计算。
    
    Parameters:
    - best_px: 当前买1（Bid）和卖1（Ask）价格
    - on_side: 当前挂单的方向 (0=Bid, 1=Ask)
    - on_px: 当前挂单价格
    - on_qty_remain: 当前剩余挂单数量
    - curr_dataset: 用于存储结果的数组，形状为 n*2，第0列为Bid侧，第1列为Ask侧
    """

    # 边界处理：买1或卖1价格无效
    bid1 = best_px[0]
    ask1 = best_px[1]
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化结果索引
    index = 0

    # Bid侧计算
    bid_idx = on_side == 0
    bid_amounts = (on_px[bid_idx] * on_qty_remain[bid_idx]) / 10000
    if bid_amounts.size > 0:
        total_amount_bid = np.sum(bid_amounts)
        if total_amount_bid > 0:
            prob_bid = bid_amounts / total_amount_bid
            prob_bid = prob_bid[prob_bid > 0]  # 过滤掉概率为0的值，避免log计算问题
            entropy_bid = -np.sum(prob_bid * np.log(prob_bid))
            curr_dataset[index, 0] = entropy_bid
        else:
            curr_dataset[index, 0] = np.nan
    else:
        curr_dataset[index, 0] = np.nan

    # Ask侧计算
    ask_idx = on_side == 1
    ask_amounts = (on_px[ask_idx] * on_qty_remain[ask_idx]) / 10000
    if ask_amounts.size > 0:
        total_amount_ask = np.sum(ask_amounts)
        if total_amount_ask > 0:
            prob_ask = ask_amounts / total_amount_ask
            prob_ask = prob_ask[prob_ask > 0]  # 过滤掉概率为0的值，避免log计算问题
            entropy_ask = -np.sum(prob_ask * np.log(prob_ask))
            curr_dataset[index, 1] = entropy_ask
        else:
            curr_dataset[index, 1] = np.nan
    else:
        curr_dataset[index, 1] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def MidPriceProximityWeightedAmount(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    按挂单价格距离中间价的百分比加权，计算 Bid 和 Ask 两侧的总加权金额。
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 计算中间价
    mid_price = (bid1 + ask1) / 2

    # 初始化行索引
    index = 0

    # Bid侧计算
    bid_idx = (on_side == 0)
    if np.any(bid_idx):
        bid_prices = on_px[bid_idx]
        bid_qty = on_qty_remain[bid_idx]
        bid_weights = 1 / (1 + np.abs(bid_prices - mid_price) / mid_price)
        bid_amounts = bid_prices * bid_qty / 10000
        curr_dataset[index, 0] = np.sum(bid_weights * bid_amounts)
    else:
        curr_dataset[index, 0] = np.nan

    # Ask侧计算
    ask_idx = (on_side == 1)
    if np.any(ask_idx):
        ask_prices = on_px[ask_idx]
        ask_qty = on_qty_remain[ask_idx]
        ask_weights = 1 / (1 + np.abs(ask_prices - mid_price) / mid_price)
        ask_amounts = ask_prices * ask_qty / 10000
        curr_dataset[index, 1] = np.sum(ask_weights * ask_amounts)
    else:
        curr_dataset[index, 1] = np.nan