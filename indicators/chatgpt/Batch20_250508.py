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
    types.int64[:],  # on_ts_org
    types.int64[:],  # on_qty_d
    types.int64[:],  # on_ts_d
    types.int64,     # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # time_ranges
    types.float64[:, :]  # curr_dataset
))
def TimeRangeOANet(best_px, on_side, on_px, on_qty_org, on_ts_org, on_qty_d, on_ts_d, ts, value_thresholds, time_ranges, curr_dataset):
    """
    计算不同挂单金额、不同时间范围内的挂单总量减去撤单总量的净值。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 原始挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - on_qty_d: 撤单量
    - on_ts_d: 撤单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为毫秒
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围），列对应 Bid 和 Ask
    
    注：
    - 金额计算采用 on_px * on_qty_org / 10000 和 on_px * on_qty_d / 10000
    - 函数会遍历所有金额阈值和时间范围组合
    - 结果为挂单金额减去撤单金额的净值
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for time_range in time_ranges:  # 遍历所有时间范围
            time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值（分钟转毫秒）
            
            # Bid 和 Ask 侧分别处理
            for side, col in [(0, 0), (1, 1)]:
                # 筛选出符合条件的大额挂单：方向匹配、金额大于阈值、时间在范围内
                order_mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= time_threshold)
                
                # 筛选出符合条件的大额撤单：方向匹配、金额大于阈值、时间在范围内
                cancel_mask = (on_side == side) & (on_px * on_qty_d / 10000 >= T) & (on_ts_d >= time_threshold)
                
                # 计算挂单总金额
                order_amount = 0
                if np.any(order_mask):
                    order_amount = np.sum(on_px[order_mask] * on_qty_org[order_mask] / 10000)
                
                # 计算撤单总金额
                cancel_amount = 0
                if np.any(cancel_mask):
                    cancel_amount = np.sum(on_px[cancel_mask] * on_qty_d[cancel_mask] / 10000)
                
                # 计算净值 = 挂单金额 - 撤单金额
                curr_dataset[index, col] = order_amount - cancel_amount
            
            index += 1