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
    types.int64,     # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # time_ranges
    types.float64[:, :]  # curr_dataset
))
def TimeRangeOAExNoon(best_px, on_side, on_px, on_qty_org, on_ts_org, ts, value_thresholds, time_ranges, curr_dataset):
    """
    计算不同挂单金额、不同时间范围内的挂单总量（排除A股午休时间）。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 当前剩余挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为分钟
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围），列对应 Bid 和 Ask
    
    注：
    - 金额计算采用 on_px * on_qty_org / 10000
    - 时间计算排除A股午休时间（11:30-13:00），复用 get_residue_time 函数
    - 函数会遍历所有金额阈值和时间范围组合
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 向量化预计算
    # 使用 get_residue_time 计算所有订单的剩余时间（毫秒）
    residue_times = get_residue_time(ts, on_ts_org)
    # breakpoint()
    
    # 向量化计算所有订单的金额
    order_amounts = on_px.astype(np.float64) * on_qty_org.astype(np.float64) / 10000.0
    
    # 创建买单和卖单的掩码
    bid_mask = (on_side == 0)
    ask_mask = (on_side == 1)
    
    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        # 预计算金额条件
        value_mask = (order_amounts >= T)
        
        for time_range in time_ranges:  # 遍历所有时间范围（分钟）
            # 将时间范围转换为毫秒，并计算时间条件
            time_range_ms = time_range * 60 * 1000
            # 订单在时间范围内：剩余时间 <= 时间范围
            time_mask = (residue_times < time_range_ms)
            
            # 组合所有条件
            base_condition = value_mask & time_mask
            
            # Bid 侧计算
            bid_condition = base_condition & bid_mask
            if np.any(bid_condition):
                curr_dataset[index, 0] = np.sum(order_amounts[bid_condition])
            else:
                curr_dataset[index, 0] = 0.0
            
            # Ask 侧计算
            ask_condition = base_condition & ask_mask
            if np.any(ask_condition):
                curr_dataset[index, 1] = np.sum(order_amounts[ask_condition])
            else:
                curr_dataset[index, 1] = 0.0
            
            index += 1


# 使用示例：
# 金额阈值列表（例如：1000, 5000, 10000 元）
# value_thresholds = np.array([1000.0, 5000.0, 10000.0], dtype=np.float64)
# 
# 时间范围列表（例如：1分钟、5分钟、15分钟、30分钟）
# time_ranges = np.array([1.0, 5.0, 15.0, 30.0], dtype=np.float64)
# 
# 结果数组：rows = len(value_thresholds) * len(time_ranges), cols = 2 (bid和ask)
# curr_dataset = np.zeros((len(value_thresholds) * len(time_ranges), 2), dtype=np.float64)