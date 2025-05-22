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
    types.int64[:],      # best_px
    types.int32[:],      # on_side
    types.int64[:],      # on_px
    types.int64[:],      # on_qty_org
    types.int64[:],      # on_ts_org
    types.int64,         # ts
    types.float64[:],    # value_thresholds
    types.float64[:],    # time_ranges
    types.float64[:],    # exclude_seconds
    types.float64[:, :]  # curr_dataset - 保持二维数组
))
def TRangeOAExSeconds(best_px, on_side, on_px, on_qty_org, on_ts_org, ts, 
                      value_thresholds, time_ranges, exclude_seconds, curr_dataset):
    """
    计算不同挂单金额、不同时间范围内的挂单总量，并为每个exclude_seconds值分别计算一个因子。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 原始挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为毫秒
    - exclude_seconds: 需要剔除的秒间隔列表，如[1, 30]表示分别计算剔除每秒和每30秒的整秒订单的因子
    - curr_dataset: 存储结果数组，行顺序为 value_thresholds -> time_ranges -> exclude_seconds 的组合，列为 Bid 和 Ask
    
    注：
    - 金额计算采用 on_px * on_qty_org / 10000
    - 函数会按顺序遍历所有组合并依次存储在二维数组中
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
            time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值，分钟转毫秒
            
            for i in range(len(exclude_seconds)):  # 遍历所有需要排除的秒间隔
                n = exclude_seconds[i]
                
                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    # 基本条件：方向匹配、金额大于阈值、时间在范围内
                    base_mask = (on_side == side) & \
                               (on_px * on_qty_org / 10000 >= T) & \
                               (on_ts_org >= time_threshold)
                    
                    if n == 0:  # 如果n=0，表示不排除任何订单
                        final_mask = base_mask
                    else:
                        # 将毫秒时间戳转换为秒
                        seconds = on_ts_org // 1000
                        # 检查是否为n秒的整数倍
                        is_exact_n_second = (seconds % n) == 0
                        # 排除整n秒的订单
                        final_mask = base_mask & ~is_exact_n_second
                    
                    if np.any(final_mask):
                        # 计算符合条件的挂单总金额
                        total_amount = np.sum(on_px[final_mask] * on_qty_org[final_mask] / 10000)
                        curr_dataset[index, col] = total_amount
                    else:
                        curr_dataset[index, col] = 0  # 无符合条件的挂单记为0
                
                index += 1  # 每处理完一个组合，索引加1

# 使用示例：
# 金额阈值列表（例如：1000, 5000, 10000 元）
# value_thresholds = np.array([1000.0, 5000.0, 10000.0], dtype=np.float64)
# 
# 时间范围列表（例如：1分钟、5分钟、30分钟）
# time_ranges = np.array([60*1000, 5*60*1000, 30*60*1000], dtype=np.int64)
# 
# 结果数组：rows = len(value_thresholds) * len(time_ranges), cols = 2 (bid和ask)
# curr_dataset = np.zeros((len(value_thresholds) * len(time_ranges), 2), dtype=np.float64)