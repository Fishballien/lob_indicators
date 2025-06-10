# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 2025

@author: Claude

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
# @njit(types.void(
#     types.int64[:],  # best_px
#     types.int32[:],  # on_side
#     types.int64[:],  # on_px
#     types.int64[:],  # on_qty_org
#     types.int64[:],  # on_qty_remain
#     types.int64[:],  # on_qty_d
#     types.int64[:],  # on_qty_t
#     types.int64[:],  # on_amt_t
#     types.int64[:],  # on_qty_t_a (主动成交量)
#     types.int64[:],  # on_amt_t_a (主动成交金额)
#     types.int64[:],  # on_qty_t_p (被动成交量)
#     types.int64[:],  # on_amt_t_p (被动成交金额)
#     types.int64[:],  # on_qty_t_n (集合竞价成交量)
#     types.int64[:],  # on_amt_t_n (集合竞价成交金额)
#     types.int64[:],  # on_ts_org
#     types.int64,     # ts
#     types.float64[:],  # value_thresholds
#     types.float64[:],  # data_types
#     types.float64[:],  # time_ranges
#     types.float64[:, :]  # curr_dataset
# ))
def TimeRangeDataTypes(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_qty_d, on_qty_t, on_amt_t,
                       on_qty_t_a, on_amt_t_a, on_qty_t_p, on_amt_t_p, on_qty_t_n, on_amt_t_n,
                       on_ts_org, ts, value_thresholds, data_types, time_ranges, curr_dataset):
    """
    计算不同数据类型、不同金额阈值、不同时间范围内的挂单金额。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 原始挂单量
    - on_qty_remain: 当前剩余挂单量
    - on_qty_d: 撤单量
    - on_qty_t: 总成交量
    - on_amt_t: 总成交金额
    - on_qty_t_a: 主动成交量
    - on_amt_t_a: 主动成交金额
    - on_qty_t_p: 被动成交量
    - on_amt_t_p: 被动成交金额
    - on_qty_t_n: 集合竞价成交量
    - on_amt_t_n: 集合竞价成交金额
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - data_types: 数据类型列表（1-5）
    - time_ranges: 时间范围列表，单位为分钟
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*数据类型*时间范围），列对应 Bid 和 Ask
    
    数据类型说明：
    1. 挂单金额：近期新挂单的所有挂单金额（on_qty_org * on_px / 10000）
    2. 留存挂单金额：近期新挂单的留存挂单金额（on_qty_remain * on_px / 10000）
    3. 撤单金额：近期新挂单已撤单金额（on_qty_d * on_px / 10000）
    4. 主动成交的原始金额：主动成交类的原始挂单金额（on_qty_t_a * on_px / 10000）
    5. 被动成交的原始金额：被动成交类的原始挂单金额（on_qty_t_p * on_px / 10000）
    
    注：
    - 所有data_type都使用相同的大单筛选条件：on_px * on_qty_org / 10000 >= T
    - 金额计算采用 quantity * on_px / 10000
    - 主动成交和被动成交现在基于新的字段 on_qty_t_a 和 on_qty_t_p
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for data_type in data_types:  # 遍历所有数据类型
            for time_range in time_ranges:  # 遍历所有时间范围
                time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值（转换为毫秒）
                
                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    # 基础条件：方向匹配、金额大于阈值、时间在范围内
                    base_mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= time_threshold)
                    
                    total_amount = 0.0
                    
                    if data_type == 1.0:  # 挂单金额
                        # 条件：on_qty_org > 0
                        mask = base_mask & (on_qty_org > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_org[mask] * on_px[mask] / 10000)
                    
                    elif data_type == 2.0:  # 留存挂单金额
                        # 条件：on_qty_remain > 0
                        mask = base_mask & (on_qty_remain > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_remain[mask] * on_px[mask] / 10000)
                    
                    elif data_type == 3.0:  # 撤单金额
                        # 条件：on_qty_d > 0
                        mask = base_mask & (on_qty_d > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_d[mask] * on_px[mask] / 10000)
                    
                    elif data_type == 4.0:  # 主动成交的原始金额
                        # 条件：on_qty_t_a > 0（有主动成交）
                        mask = base_mask & (on_qty_t_a > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_t_a[mask] * on_px[mask] / 10000)
                    
                    elif data_type == 5.0:  # 被动成交的原始金额
                        # 条件：on_qty_t_p > 0（有被动成交）
                        mask = base_mask & (on_qty_t_p > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_t_p[mask] * on_px[mask] / 10000)
                    
                    curr_dataset[index, col] = total_amount
                
                index += 1
    if ts==1547119500000:
        breakpoint()

