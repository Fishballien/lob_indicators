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
    types.int32[:],       # on_side: 挂单方向
    types.int64[:],       # on_qty_remain: 当前剩余挂单量
    types.float64[:, :]   # curr_dataset: 用于存储结果的二维数组
))
def SliceVolumeEntropy(
    on_side,              # int32[:], 挂单方向 (0 for Bid, 1 for Ask)
    on_qty_remain,        # int64[:], 当前剩余挂单量
    curr_dataset          # float64[:, :], 用于存储结果的二维数组
):
    """
    切片内量分布熵因子
    - on_side: 挂单方向 (0 for Bid, 1 for Ask)
    - on_qty_remain: 当前剩余挂单量
    - curr_dataset: 用于存储结果，二维数组，n*2 结构 (Bid 在第一列，Ask 在第二列)
    """
    # 初始化
    for side in range(2):  # 分别计算 Bid (0) 和 Ask (1)
        side_idx = (on_side == side)  # 筛选对应侧的挂单
        qty_remain_side = on_qty_remain[side_idx]

        if qty_remain_side.size == 0:  # 如果该侧无挂单，因子值置为 NaN
            curr_dataset[0, side] = np.nan
            continue

        # 计算总挂单量
        total_qty = np.sum(qty_remain_side)
        if total_qty == 0:  # 如果总挂单量为 0，因子值置为 NaN
            curr_dataset[0, side] = np.nan
            continue

        # 计算每个挂单量占比
        p = qty_remain_side / total_qty

        # 计算信息熵 H
        H = -np.sum(p * np.log(p))

        # 计算总挂单个数 N
        N = qty_remain_side.size

        # 计算归一化熵 F
        if N > 1:  # 确保 N > 1，避免 log(N) 为 0
            max_entropy = np.log(N)
            F = H / max_entropy
        else:
            F = 0.0  # 如果 N == 1，则熵为 0

        # 存入结果
        curr_dataset[0, side] = F
        
        
@njit(types.void(
    types.int64[:],       # on_px: 挂单价格
    types.int32[:],       # on_side: 挂单方向
    types.int64[:],       # on_qty_remain: 当前剩余挂单量
    types.float64[:],     # alpha: 衰减系数
    types.float64[:, :]   # curr_dataset: 用于存储结果的二维数组
))
def WeightedSliceVolumeEntropy(
    on_px,              # int64[:], 挂单价格
    on_side,            # int32[:], 挂单方向 (0 for Bid, 1 for Ask)
    on_qty_remain,      # int64[:], 当前剩余挂单量
    alpha,              # float64[:], 衰减系数
    curr_dataset        # float64[:, :], 用于存储结果的二维数组
):
    """
    切片内量分布加权熵因子
    - on_px: 挂单价格
    - on_side: 挂单方向 (0 for Bid, 1 for Ask)
    - on_qty_remain: 当前剩余挂单量
    - alpha: 衰减系数，影响权重计算
    - curr_dataset: 用于存储结果，二维数组，n*2 结构 (Bid 在第一列，Ask 在第二列)
    """
    for a_idx, alpha_val in enumerate(alpha):  # 遍历衰减系数
        for side in range(2):  # 分别计算 Bid (0) 和 Ask (1)
            side_idx = (on_side == side)  # 筛选对应侧的挂单
            px_side = on_px[side_idx]
            qty_side = on_qty_remain[side_idx]

            if qty_side.size == 0:  # 如果该侧无挂单，因子值置为 NaN
                curr_dataset[a_idx, side] = np.nan
                continue

            # 计算价格中心 (加权均值)
            total_qty = np.sum(qty_side)
            if total_qty == 0:  # 如果总挂单量为 0，因子值置为 NaN
                curr_dataset[a_idx, side] = np.nan
                continue

            px_center = np.sum(px_side * qty_side) / total_qty

            # 计算权重
            weights = np.exp(-alpha_val * np.abs(px_side - px_center))

            # 计算加权比例
            weighted_qty = weights * qty_side
            total_weighted_qty = np.sum(weighted_qty)
            if total_weighted_qty == 0:  # 如果总加权量为 0，因子值置为 NaN
                curr_dataset[a_idx, side] = np.nan
                continue

            p = weighted_qty / total_weighted_qty

            # 计算加权熵 H
            H = -np.sum(p * np.log(p))

            # 计算总挂单个数 N
            N = qty_side.size

            # 计算归一化熵 F
            if N > 1:  # 确保 N > 1，避免 log(N) 为 0
                max_entropy = np.log(N)
                F = H / max_entropy
            else:
                F = 0.0  # 如果 N == 1，则熵为 0

            # 存入结果
            curr_dataset[a_idx, side] = F
            
            
@njit(types.void(
    types.int64[:],       # on_px: 挂单价格
    types.int32[:],       # on_side: 挂单方向
    types.int64[:],       # on_qty_remain: 当前剩余挂单量
    types.float64[:, :]   # curr_dataset: 用于存储结果的二维数组
))
def PriceWeightedVolumeCenterShift(
    on_px,              # int64[:], 挂单价格
    on_side,            # int32[:], 挂单方向 (0 for Bid, 1 for Ask)
    on_qty_remain,      # int64[:], 当前剩余挂单量
    curr_dataset        # float64[:, :], 用于存储结果的二维数组
):
    """
    切片内价格加权量重心偏移因子
    - on_px: 挂单价格
    - on_side: 挂单方向 (0 for Bid, 1 for Ask)
    - on_qty_remain: 当前剩余挂单量
    - curr_dataset: 用于存储结果，二维数组，n*2 结构 (Bid 在第一列，Ask 在第二列)
    """
    for side in range(2):  # 分别计算 Bid (0) 和 Ask (1)
        side_idx = (on_side == side)  # 筛选对应侧的挂单
        px_side = on_px[side_idx]
        qty_side = on_qty_remain[side_idx]

        if qty_side.size == 0:  # 如果该侧无挂单，因子值置为 NaN
            curr_dataset[0, side] = np.nan
            continue

        # 计算价格中心 P_center
        total_qty = np.sum(qty_side)
        if total_qty == 0:  # 如果总挂单量为 0，因子值置为 NaN
            curr_dataset[0, side] = np.nan
            continue

        P_center = np.sum(px_side * qty_side) / total_qty

        # 计算量重心 P_weight
        qty_squared = qty_side ** 2
        total_qty_squared = np.sum(qty_squared)
        if total_qty_squared == 0:  # 如果总平方权重为 0，因子值置为 NaN
            curr_dataset[0, side] = np.nan
            continue

        P_weight = np.sum(px_side * qty_squared) / total_qty_squared

        # 计算偏移因子 F
        if P_center == 0:  # 避免分母为 0 的情况
            curr_dataset[0, side] = np.nan
        else:
            F = (P_weight - P_center) / P_center
            curr_dataset[0, side] = F
            
            
@njit(types.void(
    types.int64[:],       # on_px: 挂单价格
    types.int32[:],       # on_side: 挂单方向
    types.int64[:],       # on_qty_remain: 当前剩余挂单量
    types.float64[:],     # delta_p_pct: 中间价百分比的价格变化
    types.float64[:, :]   # curr_dataset: 用于存储结果的二维数组
))
def LiquidityElasticity(
    on_px,               # int64[:], 挂单价格
    on_side,             # int32[:], 挂单方向 (0 for Bid, 1 for Ask)
    on_qty_remain,       # int64[:], 当前剩余挂单量
    delta_p_pct,         # float64[:], 中间价百分比的价格变化
    curr_dataset         # float64[:, :], 用于存储结果的二维数组
):
    """
    切片内流动性弹性因子
    - on_px: 挂单价格
    - on_side: 挂单方向 (0 for Bid, 1 for Ask)
    - on_qty_remain: 当前剩余挂单量
    - delta_p_pct: 中间价百分比的价格变化
    - curr_dataset: 用于存储结果，二维数组，n*2 结构 (Bid 在第一列，Ask 在第二列)
    """
    for dp_idx, delta_p_factor in enumerate(delta_p_pct):  # 遍历 delta_p 参数
        for side in range(2):  # 分别计算 Bid (0) 和 Ask (1)
            side_idx = (on_side == side)  # 筛选对应侧的挂单
            px_side = on_px[side_idx]
            qty_side = on_qty_remain[side_idx]

            if qty_side.size == 0:  # 如果该侧无挂单，因子值置为 NaN
                curr_dataset[dp_idx, side] = np.nan
                continue

            # 计算价格中心 P_center
            total_qty = np.sum(qty_side)
            if total_qty == 0:  # 如果总挂单量为 0，因子值置为 NaN
                curr_dataset[dp_idx, side] = np.nan
                continue

            P_center = np.sum(px_side * qty_side) / total_qty

            # 计算价格偏移 Delta P
            delta_p = P_center * delta_p_factor

            # 计算高价方向弹性 E_up
            up_mask = px_side >= P_center + delta_p
            px_up = px_side[up_mask]
            qty_up = qty_side[up_mask]
            if qty_up.size > 0:
                E_up = np.sum(qty_up * (px_up - P_center))
            else:
                E_up = 0.0

            # 计算低价方向弹性 E_down
            down_mask = px_side <= P_center - delta_p
            px_down = px_side[down_mask]
            qty_down = qty_side[down_mask]
            if qty_down.size > 0:
                E_down = np.sum(qty_down * (P_center - px_down))
            else:
                E_down = 0.0

            # 计算流动性弹性因子 F
            total_elasticity = E_up + E_down
            if total_elasticity == 0:  # 避免分母为 0
                F = 0.0
            else:
                F = (E_up - E_down) / total_elasticity

            # 存入结果
            curr_dataset[dp_idx, side] = F
