# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 2025

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
def FastCancelAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_qty_d, on_amt_t,
                     on_qty_t_a, on_amt_t_a, on_qty_t_p, on_amt_t_p, on_qty_t_n, on_amt_t_n,
                     on_ts_org, on_ts_d, ts, value_thresholds, cancel_speeds, time_windows, curr_dataset):
    """
    计算不同时间窗口内、不同速度撤单的金额统计（向量化实现）。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 原始挂单量
    - on_qty_remain: 当前剩余挂单量
    - on_qty_d: 撤单量
    - on_amt_t: 总成交金额
    - on_qty_t_a: 主动成交量
    - on_amt_t_a: 主动成交金额
    - on_qty_t_p: 被动成交量
    - on_amt_t_p: 被动成交金额
    - on_qty_t_n: 集合竞价成交量
    - on_amt_t_n: 集合竞价成交金额
    - on_ts_org: 挂单时间戳（毫秒）
    - on_ts_d: 撤单时间戳（毫秒）
    - ts: 当前时间戳（毫秒）
    - value_thresholds: 金额阈值列表，单位为万元
    - cancel_speeds: 撤单速度阈值列表，单位为秒
    - time_windows: 统计时间窗口列表，单位为分钟
    - curr_dataset: 存储结果数组，行对应参数组合，列对应 Bid 和 Ask
    
    输出：
    curr_dataset[i, 0] = 买单侧快速撤单金额
    curr_dataset[i, 1] = 卖单侧快速撤单金额
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 预计算基础条件
    # 有效订单基础mask：有挂单时间、撤单时间、撤单量
    valid_orders = (on_ts_org > 0) & (on_ts_d > 0) & (on_qty_d > 0)
    
    # 预计算撤单持续时间（毫秒）
    cancel_duration = np.where(valid_orders, on_ts_d - on_ts_org, np.inf)
    
    # 预计算订单金额（万元）
    order_amounts = on_px * on_qty_org / 10000
    
    # 预计算撤单金额（万元）
    cancel_amounts = on_px * on_qty_d / 10000
    
    index = 0
    for T in value_thresholds:  # 遍历金额阈值
        for cancel_speed in cancel_speeds:  # 遍历撤单速度（秒）
            cancel_speed_ms = cancel_speed * 1000  # 转换为毫秒
            
            for time_window in time_windows:  # 遍历统计窗口（分钟）
                window_start = ts - time_window * 60 * 1000  # 统计窗口开始时间
                
                # 构建通用mask
                base_mask = (
                    valid_orders &                          # 有效订单
                    (order_amounts >= T) &                  # 金额阈值
                    (on_ts_org >= window_start) &           # 时间窗口
                    (cancel_duration <= cancel_speed_ms)    # 撤单速度
                )
                
                # 买单侧（side=0）
                bid_mask = base_mask & (on_side == 0)
                curr_dataset[index, 0] = np.sum(cancel_amounts[bid_mask]) if np.any(bid_mask) else 0.0
                
                # 卖单侧（side=1）
                ask_mask = base_mask & (on_side == 1)
                curr_dataset[index, 1] = np.sum(cancel_amounts[ask_mask]) if np.any(ask_mask) else 0.0
                
                index += 1