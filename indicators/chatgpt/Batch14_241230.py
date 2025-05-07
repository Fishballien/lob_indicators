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
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float32[:, :]  # curr_dataset
))
def VolumeDirectionalIndex(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, curr_dataset):
    """
    计算成交量方向性指数（Volume Directional Index）
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - curr_dataset: 存储结果数组，行对应不同的因子，列对应 Bid 和 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化方向性分量和总成交量
    bid_direction_x = 0.0
    bid_direction_y = 0.0
    ask_direction_x = 0.0
    ask_direction_y = 0.0
    total_bid_volume = 0.0
    total_ask_volume = 0.0

    # 计算 Bid 侧的方向性
    for i in range(len(on_side)):
        if on_side[i] == 0:  # Bid 侧
            distance = (bid1 - on_px[i]) / 10000  # 价格与买一价的距离（单位：元）
            angle = np.arctan(distance)  # 计算角度（以买一价为基准）
            volume = on_qty_remain[i]
            
            # 将成交量分解到 x 和 y 分量
            bid_direction_x += volume * np.cos(angle)
            bid_direction_y += volume * np.sin(angle)
            total_bid_volume += volume

    # 计算 Ask 侧的方向性
    for i in range(len(on_side)):
        if on_side[i] == 1:  # Ask 侧
            distance = (on_px[i] - ask1) / 10000  # 价格与卖一价的距离（单位：元）
            angle = np.arctan(distance)  # 计算角度（以卖一价为基准）
            volume = on_qty_remain[i]
            
            # 将成交量分解到 x 和 y 分量
            ask_direction_x += volume * np.cos(angle)
            ask_direction_y += volume * np.sin(angle)
            total_ask_volume += volume

    # 计算方向性指数
    if total_bid_volume > 0:
        bid_directionality_index = (bid_direction_x ** 2 + bid_direction_y ** 2) / (total_bid_volume ** 2)
    else:
        bid_directionality_index = np.nan

    if total_ask_volume > 0:
        ask_directionality_index = (ask_direction_x ** 2 + ask_direction_y ** 2) / (total_ask_volume ** 2)
    else:
        ask_directionality_index = np.nan

    # 将结果填入当前数据集
    curr_dataset[:, 0] = bid_directionality_index
    curr_dataset[:, 1] = ask_directionality_index
    
    
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float32[:, :]  # curr_dataset
))
def VolumeConcentrationIndex(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, curr_dataset):
    """
    计算成交量集中度指数因子（Volume Concentration Index）
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - curr_dataset: 存储结果数组，行对应不同的因子，列对应 Bid 和 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 检查买卖一价格是否有效
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 计算成交量集中度
    bid_concentration = 0.0
    total_bid_volume = 0.0

    # 计算 Bid 侧的集中度
    for i in range(len(on_side)):
        if on_side[i] == 0:  # Bid 侧
            distance = (bid1 - on_px[i]) / 10000  # 距离
            angle = (i * np.pi) / len(on_side)  # 角度计算
            volume = on_qty_remain[i]
            weight = volume / np.sum(on_qty_remain)
            bid_concentration += weight * np.cos(angle)
            total_bid_volume += volume

    ask_concentration = 0.0
    total_ask_volume = 0.0

    # 计算 Ask 侧的集中度
    for i in range(len(on_side)):
        if on_side[i] == 1:  # Ask 侧
            distance = (on_px[i] - ask1) / 10000  # 距离
            angle = (i * np.pi) / len(on_side)  # 角度计算
            volume = on_qty_remain[i]
            weight = volume / np.sum(on_qty_remain)
            ask_concentration += weight * np.cos(angle)
            total_ask_volume += volume

    # 填充结果
    curr_dataset[:, 0] = bid_concentration
    curr_dataset[:, 1] = ask_concentration


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float32[:, :]  # curr_dataset
))
def PriceDepthReversalIndex(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, curr_dataset):
    """
    计算价格深度反向性指数因子（Price Depth Reversal Index）
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - curr_dataset: 存储结果数组，行对应不同的因子，列对应 Bid 和 Ask
    """
    # 初始化买卖深度的成交量
    total_buy_volume = 0.0
    total_sell_volume = 0.0
    reversal_sum = 0.0
    total_volume = 0.0

    # 计算 Bid 侧的反向性
    for i in range(len(on_side)):
        if on_side[i] == 0:  # Bid 侧
            volume = on_qty_remain[i]
            total_buy_volume += volume
            reversal_sum += volume * (-1) ** i  # 乘以(-1)^i 来区分买卖双方

    # 计算 Ask 侧的反向性
    for i in range(len(on_side)):
        if on_side[i] == 1:  # Ask 侧
            volume = on_qty_remain[i]
            total_sell_volume += volume
            reversal_sum += volume * (-1) ** i  # 乘以(-1)^i 来区分买卖双方

    # 总成交量
    total_volume = total_buy_volume + total_sell_volume

    # 如果总成交量为 0，则返回 NaN
    if total_volume == 0:
        curr_dataset[:, :] = np.nan
        return

    # 计算反向性指数（RDI）
    rdi = reversal_sum / total_volume

    # 填充结果
    curr_dataset[:, 0] = rdi  # 可以填充在 Bid 侧的 RDI
    curr_dataset[:, 1] = rdi  # 可以填充在 Ask 侧的 RDI
