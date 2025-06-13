# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 2025

@author: Assistant

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
def OrderStatusAnalysis(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_qty_d, on_qty_t, on_amt_t,
                       on_qty_t_a, on_amt_t_a, on_qty_t_p, on_amt_t_p, on_qty_t_n, on_amt_t_n,
                       on_ts_org, ts, value_thresholds, order_types, time_ranges, curr_dataset):
    """
    计算不同订单状态、不同金额阈值、不同时间范围内的挂单金额。
    
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
    - order_types: 订单类型列表（1-5）
    - time_ranges: 时间范围列表，单位为分钟
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*订单类型*时间范围），列对应 Bid 和 Ask
    
    订单类型说明：
    1. 所有挂单金额：近期新挂单的所有挂单金额（on_qty_org * on_px / 10000）
    2. 发生了部分成交的撤单：原始挂单有成交且有撤单（on_qty_t > 0 and on_qty_d > 0）
    3. 未成交的完整撤单：原始挂单无成交但完全撤单（on_qty_t == 0 and on_qty_d == on_qty_org）
    4. 发生了部分成交的剩余挂单：原始挂单有成交且有剩余（on_qty_t > 0 and on_qty_remain > 0）
    5. 未成交的完整剩余挂单：原始挂单无成交且完全剩余（on_qty_t == 0 and on_qty_remain == on_qty_org）
    
    注：
    - 所有order_type都使用相同的大单筛选条件：on_px * on_qty_org / 10000 >= T
    - 对于类型2和3，金额计算使用撤单量：on_qty_d * on_px / 10000
    - 对于类型4和5，金额计算使用剩余量：on_qty_remain * on_px / 10000
    - 对于类型1，金额计算使用原始量：on_qty_org * on_px / 10000
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for order_type in order_types:  # 遍历所有订单类型
            for time_range in time_ranges:  # 遍历所有时间范围
                time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值（转换为毫秒）
                
                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    # 基础条件：方向匹配、金额大于阈值、时间在范围内
                    base_mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= time_threshold)
                    
                    total_amount = 0.0
                    
                    if order_type == 1.0:  # 所有挂单金额
                        # 条件：on_qty_org > 0
                        mask = base_mask & (on_qty_org > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_org[mask] * on_px[mask] / 10000)
                    
                    elif order_type == 2.0:  # 发生了部分成交的撤单
                        # 条件：on_qty_t > 0 and on_qty_d > 0
                        mask = base_mask & (on_qty_t > 0) & (on_qty_d > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_d[mask] * on_px[mask] / 10000)
                    
                    elif order_type == 3.0:  # 未成交的完整撤单
                        # 条件：on_qty_t == 0 and on_qty_d == on_qty_org
                        mask = base_mask & (on_qty_t == 0) & (on_qty_d == on_qty_org) & (on_qty_org > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_d[mask] * on_px[mask] / 10000)
                    
                    elif order_type == 4.0:  # 发生了部分成交的剩余挂单
                        # 条件：on_qty_t > 0 and on_qty_remain > 0
                        mask = base_mask & (on_qty_t > 0) & (on_qty_remain > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_remain[mask] * on_px[mask] / 10000)
                    
                    elif order_type == 5.0:  # 未成交的完整剩余挂单
                        # 条件：on_qty_t == 0 and on_qty_remain == on_qty_org
                        mask = base_mask & (on_qty_t == 0) & (on_qty_remain == on_qty_org) & (on_qty_org > 0)
                        if np.any(mask):
                            total_amount = np.sum(on_qty_remain[mask] * on_px[mask] / 10000)
                    
                    curr_dataset[index, col] = total_amount
                
                index += 1