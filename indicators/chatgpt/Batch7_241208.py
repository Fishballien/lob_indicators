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
@njit(
    types.void(
        types.int64[:],  # best_px (买一/卖一价格，长度为2的数组)
        types.int32[:],  # on_side (挂单方向，0为Bid，1为Ask)
        types.int64[:],  # on_px (挂单价格)
        types.int64[:],  # on_qty_remain (挂单剩余数量)
        types.float64[:, :]  # curr_dataset (用于存储结果的二维数组)
    )
)
def OrderDensity(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单数量密度因子：
    计算单侧挂单数量与价格区间宽度的比值。

    参数说明：
    - best_px: 当前买一和卖一价格 (int64[:])，长度为2
    - on_side: 当前挂单方向 (int32[:])，0为买单，1为卖单
    - on_px: 当前挂单价格 (int64[:])
    - on_qty_remain: 当前挂单剩余数量 (int64[:])
    - curr_dataset: 用于存储计算结果的二维数组 (float64[:, :])，
        - 第0列存储Bid侧结果
        - 第1列存储Ask侧结果
        - 只有1行，因为该因子无参数遍历
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid 侧计算
    bid_idx = on_side == 0
    if np.any(bid_idx):  # 如果存在 Bid 侧数据
        bid_px = on_px[bid_idx]
        bid_qty = on_qty_remain[bid_idx]
        price_range = np.max(bid_px) - np.min(bid_px)
        if price_range > 0:  # 避免除以零
            curr_dataset[0, 0] = np.sum(bid_qty) / price_range
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = 0  # Bid 无挂单时，数量记为 0

    # Ask 侧计算
    ask_idx = on_side == 1
    if np.any(ask_idx):  # 如果存在 Ask 侧数据
        ask_px = on_px[ask_idx]
        ask_qty = on_qty_remain[ask_idx]
        price_range = np.max(ask_px) - np.min(ask_px)
        if price_range > 0:  # 避免除以零
            curr_dataset[0, 1] = np.sum(ask_qty) / price_range
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = 0  # Ask 无挂单时，数量记为 0


@njit(
    types.void(
        types.int64[:],  # best_px (买一/卖一价格，长度为2的数组)
        types.int32[:],  # on_side (挂单方向，0为Bid，1为Ask)
        types.int64[:],  # on_px (挂单价格)
        types.int64[:],  # on_qty_remain (挂单剩余数量)
        types.float64[:, :]  # curr_dataset (用于存储结果的二维数组)
    )
)
def OrderPriceImpact(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单价格冲击因子：
    计算单侧挂单价格与挂单数量的 Spearman 相关系数。

    参数：
    - best_px: 买一卖一价格 (int64[:])
    - on_side: 挂单方向 (int32[:])
    - on_px: 挂单价格 (int64[:])
    - on_qty_remain: 挂单剩余数量 (int64[:])
    - curr_dataset: 存储计算结果的二维数组 (float64[:, :])
        - 第0列存储Bid侧结果
        - 第1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Spearman 相关系数计算函数
    def spearman_correlation(x, y):
        if len(x) <= 1 or len(y) <= 1:  # 数据量不足
            return np.nan

        # 排序并计算排名
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))

        # 手动计算协方差和标准差
        mean_x = np.mean(rank_x)
        mean_y = np.mean(rank_y)
        cov_xy = np.mean((rank_x - mean_x) * (rank_y - mean_y))  # 手工协方差计算
        std_x = np.sqrt(np.mean((rank_x - mean_x) ** 2))
        std_y = np.sqrt(np.mean((rank_y - mean_y) ** 2))

        if std_x == 0 or std_y == 0:
            return np.nan  # 避免除以零

        return cov_xy / (std_x * std_y)

    # Bid 侧计算
    bid_idx = on_side == 0
    if np.any(bid_idx):  # 如果存在 Bid 侧数据
        bid_px = on_px[bid_idx]
        bid_qty = on_qty_remain[bid_idx]
        if len(bid_px) > 1:  # 确保数据量足够
            curr_dataset[0, 0] = spearman_correlation(bid_px, bid_qty)
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = np.nan  # 无数据

    # Ask 侧计算
    ask_idx = on_side == 1
    if np.any(ask_idx):  # 如果存在 Ask 侧数据
        ask_px = on_px[ask_idx]
        ask_qty = on_qty_remain[ask_idx]
        if len(ask_px) > 1:  # 确保数据量足够
            curr_dataset[0, 1] = spearman_correlation(ask_px, ask_qty)
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = np.nan  # 无数据


@njit(
    types.void(
        types.int64[:],  # best_px (买一/卖一价格，长度为2的数组)
        types.int32[:],  # on_side (挂单方向，0为Bid，1为Ask)
        types.int64[:],  # on_px (挂单价格)
        types.int64[:],  # on_qty_remain (挂单剩余数量)
        types.float64[:, :]  # curr_dataset (用于存储结果的二维数组)
    )
)
def OrderResidualGradient(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    挂单残量梯度因子：
    计算单侧挂单剩余数量随价格变化的加权梯度。

    参数：
    - best_px: 买一卖一价格 (int64[:])
    - on_side: 挂单方向 (int32[:])
    - on_px: 挂单价格 (int64[:])
    - on_qty_remain: 挂单剩余数量 (int64[:])
    - curr_dataset: 存储计算结果的二维数组 (float64[:, :])
        - 第0列存储Bid侧结果
        - 第1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid 侧计算
    bid_idx = on_side == 0
    if np.any(bid_idx):  # 如果存在 Bid 侧数据
        bid_px = on_px[bid_idx]
        bid_qty = on_qty_remain[bid_idx]
        mu_px = np.mean(bid_px)  # 计算均值价格
        numerator = np.sum(bid_qty * (bid_px - mu_px))
        denominator = np.sum(np.abs(bid_px - mu_px))
        if denominator > 0:
            curr_dataset[0, 0] = numerator / denominator
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = np.nan  # 无数据

    # Ask 侧计算
    ask_idx = on_side == 1
    if np.any(ask_idx):  # 如果存在 Ask 侧数据
        ask_px = on_px[ask_idx]
        ask_qty = on_qty_remain[ask_idx]
        mu_px = np.mean(ask_px)  # 计算均值价格
        numerator = np.sum(ask_qty * (ask_px - mu_px))
        denominator = np.sum(np.abs(ask_px - mu_px))
        if denominator > 0:
            curr_dataset[0, 1] = numerator / denominator
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = np.nan  # 无数据

        curr_dataset[0, 1] = 0  # Ask 无挂单时，数量记为 0


@njit(
    types.void(
        types.int64[:],  # best_px (买一/卖一价格，长度为2的数组)
        types.int32[:],  # on_side (挂单方向，0为Bid，1为Ask)
        types.int64[:],  # on_px (挂单价格)
        types.float64[:, :]  # curr_dataset (用于存储结果的二维数组)
    )
)
def OrderPriceSkewness(best_px, on_side, on_px, curr_dataset):
    """
    挂单分布不对称性因子：
    计算挂单价格分布相对于价格中值的偏度。

    参数：
    - best_px: 买一卖一价格 (int64[:])
    - on_side: 挂单方向 (int32[:])
    - on_px: 挂单价格 (int64[:])
    - curr_dataset: 存储计算结果的二维数组 (float64[:, :])
        - 第0列存储Bid侧结果
        - 第1列存储Ask侧结果
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid 侧计算
    bid_idx = on_side == 0
    if np.any(bid_idx):  # 如果存在 Bid 侧数据
        bid_px = on_px[bid_idx]
        N = len(bid_px)
        mu_px = np.mean(bid_px)  # 均值价格
        diff = bid_px - mu_px
        numerator = np.mean(diff**3)  # 分子部分
        denominator = (np.mean(diff**2))**1.5  # 分母部分
        if denominator > 0:
            curr_dataset[0, 0] = numerator / denominator
        else:
            curr_dataset[0, 0] = np.nan
    else:
        curr_dataset[0, 0] = np.nan  # 无数据

    # Ask 侧计算
    ask_idx = on_side == 1
    if np.any(ask_idx):  # 如果存在 Ask 侧数据
        ask_px = on_px[ask_idx]
        N = len(ask_px)
        mu_px = np.mean(ask_px)  # 均值价格
        diff = ask_px - mu_px
        numerator = np.mean(diff**3)  # 分子部分
        denominator = (np.mean(diff**2))**1.5  # 分母部分
        if denominator > 0:
            curr_dataset[0, 1] = numerator / denominator
        else:
            curr_dataset[0, 1] = np.nan
    else:
        curr_dataset[0, 1] = np.nan  # 无数据