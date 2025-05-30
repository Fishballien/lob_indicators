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
    types.float64[:],  # alpha_values (power decay parameters)
    types.float64[:, :]  # curr_dataset
))
def TimeRangeOAW(best_px, on_side, on_px, on_qty_org, on_ts_org, ts, 
                 value_thresholds, time_ranges, alpha_values, curr_dataset):
    """
    计算不同挂单金额、不同时间范围、不同权重衰减参数下的加权挂单总量。
    使用向量化运算提高性能。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 当前剩余挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为分钟
    - alpha_values: power衰减参数列表，权重 = (1-t)^alpha，其中t为归一化时间差
    - curr_dataset: 存储结果数组，行对应参数组合，列对应 Bid 和 Ask
    
    结果存储顺序：
    - 外层循环：value_thresholds
    - 中层循环：time_ranges  
    - 内层循环：alpha_values
    - 每行存储：[bid_weighted_amount, ask_weighted_amount]
    
    注：
    - 金额计算采用 on_px * on_qty_org / 10000
    - 时间权重采用 power 衰减：weight = (1 - normalized_time_diff)^alpha
    - normalized_time_diff = (ts - on_ts_org) / (time_range * 1000 * 60)
    - 超出时间范围的挂单权重为0
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 预计算所有挂单的金额 (向量化)
    amounts = on_px.astype(np.float64) * on_qty_org.astype(np.float64) / 10000.0
    
    # 创建买卖单掩码 (向量化)
    bid_mask = on_side == 0
    ask_mask = on_side == 1
    
    index = 0
    
    # 外层循环：金额阈值
    for T in value_thresholds:
        # 金额过滤掩码 (向量化)
        amount_mask = amounts >= T
        
        # 中层循环：时间范围
        for time_range in time_ranges:
            time_range_ms = time_range * 1000.0 * 60.0  # 转换为毫秒
            time_threshold = ts - time_range_ms
            
            # 时间过滤掩码 (向量化)
            time_mask = on_ts_org >= time_threshold
            
            # 计算归一化时间差 (向量化)
            time_diffs = (ts - on_ts_org.astype(np.float64)) / time_range_ms
            # 确保时间差在[0,1]范围内，超出范围的设为无效值
            valid_time_mask = (time_diffs >= 0.0) & (time_diffs <= 1.0)
            
            # 内层循环：alpha参数
            for alpha in alpha_values:
                
                # 计算power衰减权重 (向量化)
                weights = np.where(valid_time_mask, (1.0 - time_diffs) ** alpha, 0.0)
                
                # 计算加权金额 (向量化)
                weighted_amounts = amounts * weights
                
                # 组合所有过滤条件
                bid_final_mask = bid_mask & amount_mask & time_mask & valid_time_mask
                ask_final_mask = ask_mask & amount_mask & time_mask & valid_time_mask
                
                # 计算Bid和Ask的加权总金额 (向量化求和)
                bid_total = np.sum(weighted_amounts * bid_final_mask.astype(np.float64))
                ask_total = np.sum(weighted_amounts * ask_final_mask.astype(np.float64))
                
                # 存储结果
                curr_dataset[index, 0] = bid_total
                curr_dataset[index, 1] = ask_total
                
                index += 1

# 使用示例：
# value_thresholds = np.array([1000.0, 5000.0], dtype=np.float64)  # 2个金额阈值
# time_ranges = np.array([1.0, 5.0], dtype=np.float64)            # 2个时间范围(分钟)
# alpha_values = np.array([0.5, 1.0, 2.0], dtype=np.float64)      # 3个alpha参数
# 
# 结果数组：rows = 2*2*3 = 12, cols = 2 (bid和ask)
# curr_dataset = np.zeros((len(value_thresholds) * len(time_ranges) * len(alpha_values), 2), dtype=np.float64)
#
# 性能优化点：
# 1. 预计算所有金额 (amounts)
# 2. 预计算买卖单掩码 (bid_mask, ask_mask)
# 3. 向量化时间差计算和权重计算
# 4. 向量化掩码组合和求和操作
# 5. 避免内层数据遍历循环
#
# Alpha参数含义：
# - α = 0.5：平方根衰减，前期衰减快
# - α = 1.0：线性衰减  
# - α = 2.0：二次衰减，后期衰减快
# - α越大，近期权重保持越久，远期衰减越快