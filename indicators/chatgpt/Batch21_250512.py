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
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # decay_list
    types.float64[:],  # filter_minutes
    types.float64[:, :]  # curr_dataset
))
def ValueTimeDecayOARmAuc(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, value_thresholds, decay_list, filter_minutes, curr_dataset):
    """
    计算筛选初始挂单量并对挂单时间做衰减的挂单金额因子。
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 初始挂单量
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - value_thresholds: 大单金额筛选阈值
    - decay_list: 时间衰减参数列表
    - filter_minutes: 筛选分钟数列表（如[25,30]表示分别计算9:25和9:30后的单）
    - curr_dataset: 存储结果数组，行对应不同参数组合，列对应 Bid 和 Ask
    """
    time_buckets = [10 * 1000, 60 * 1000, 10 * 60 * 1000, 30 * 60 * 1000]
    num_buckets = len(time_buckets) + 1

    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 计算当天0点的时间戳
    day_start_ts = (ts // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)

    index = 0
    for T in value_thresholds:
        for decay in decay_list:
            for filter_minute in filter_minutes:
                # 转换filter_minute为int64类型
                filter_minute_int = np.int64(filter_minute)
                
                # 计算9:filter_minute的时间戳
                filter_ts = day_start_ts + (9 * 60 + filter_minute_int) * 60 * 1000
                
                # 生成每层的权重
                weights = 1 - decay * np.arange(num_buckets)
                weights = np.maximum(weights, 0)
                if np.sum(weights) > 0:
                    weights /= np.sum(weights)

                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    # 添加时间筛选条件
                    mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= filter_ts)
                    if np.any(mask):
                        valid_on_px = on_px[mask]
                        valid_on_qty_remain = on_qty_remain[mask]
                        valid_on_ts_org = on_ts_org[mask]
                        time_deltas = ts - valid_on_ts_org

                        # 初始化每个时间段的金额总和
                        bucket_amounts = np.zeros(num_buckets, dtype=np.float64)

                        # 分时间段计算金额
                        for i, t_bound in enumerate(time_buckets):
                            in_bucket = time_deltas < t_bound
                            bucket_amounts[i] = np.sum(valid_on_px[in_bucket] * valid_on_qty_remain[in_bucket] / 10000)
                            time_deltas[in_bucket] = np.inf

                        # 剩余部分为最高时间段
                        bucket_amounts[-1] = np.sum(valid_on_px[time_deltas != np.inf] * valid_on_qty_remain[time_deltas != np.inf] / 10000)

                        # 加权计算
                        weighted_sum = np.sum(bucket_amounts * weights)
                        curr_dataset[index, col] = weighted_sum
                    else:
                        curr_dataset[index, col] = 0  # 无有效挂单记为0

                index += 1
            
            
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # filter_minutes
    types.float64[:, :]  # curr_dataset
))
def LargeOARmAuc(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, value_thresholds, filter_minutes, curr_dataset):
    """
    LargeOrderAmountByValue 因子计算函数：统计满足金额阈值的大单挂单金额总量。
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 初始挂单量
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - value_thresholds: 大单金额阈值
    - filter_minutes: 筛选分钟数列表（如[25,30]表示分别计算9:25和9:30后的单）
    - curr_dataset: 存储结果数组
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 计算当天0点的时间戳
    day_start_ts = (ts // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)

    index = 0
    for T in value_thresholds:
        for filter_minute in filter_minutes:
            # 转换filter_minute为int64类型
            filter_minute_int = np.int64(filter_minute)
            
            # 计算9:filter_minute的时间戳
            filter_ts = day_start_ts + (9 * 60 + filter_minute_int) * 60 * 1000
            
            # Bid侧计算 - 添加时间筛选条件
            bid_idx = (on_side == 0) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= filter_ts)
            if np.any(bid_idx):  # 如果有满足条件的数据
                curr_dataset[index, 0] = np.sum(on_px[bid_idx] * on_qty_remain[bid_idx] / 10000)
            else:
                curr_dataset[index, 0] = 0  # 没有挂单金额则记为0

            # Ask侧计算 - 添加时间筛选条件
            ask_idx = (on_side == 1) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= filter_ts)
            if np.any(ask_idx):  # 如果有满足条件的数据
                curr_dataset[index, 1] = np.sum(on_px[ask_idx] * on_qty_remain[ask_idx] / 10000)
            else:
                curr_dataset[index, 1] = 0  # 没有挂单金额则记为0

            index += 1