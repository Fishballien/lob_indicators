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
    types.float32[:, :]  # curr_dataset
))
def ValueTimeDecayOrderAmount(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, value_thresholds, decay_list, curr_dataset):
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

    index = 0
    for T in value_thresholds:
        for decay in decay_list:
            # 生成每层的权重
            weights = 1 - decay * np.arange(num_buckets)
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)

            # Bid 和 Ask 侧分别处理
            for side, col in [(0, 0), (1, 1)]:
                mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T)
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
                        # time_deltas[in_bucket] = np.inf

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
    types.float64[:],  # T_list
    types.float32[:, :]  # curr_dataset
))
def LinearDecayLargeOrderProportionByValue(best_px, on_side, on_px, on_qty_org, on_qty_remain, on_ts_org, ts, value_thresholds, T_list, curr_dataset):
    """
    按金额筛选的大单挂单线性时间衰减占比因子计算函数
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 初始挂单量
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - value_thresholds: 挂单金额阈值列表
    - T_list: 线性衰减时间窗口（毫秒）列表
    - curr_dataset: 存储结果数组，行对应参数组合，列对应 Bid 和 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for value_thres in value_thresholds:
        for T in T_list:
            for side, col in [(0, 0), (1, 1)]:
                mask = on_side == side
                if np.any(mask):
                    valid_on_px = on_px[mask]
                    valid_on_qty_org = on_qty_org[mask]
                    valid_on_qty_remain = on_qty_remain[mask]
                    valid_on_ts_org = on_ts_org[mask]

                    # 计算挂单金额
                    order_values = valid_on_px * valid_on_qty_org / 10000  # 挂单金额 (万元)

                    # 筛选大单金额
                    large_order_mask = order_values > value_thres

                    # 计算时间衰减权重
                    time_diff = ts - valid_on_ts_org
                    weights = np.maximum(1 - time_diff / T, 0)

                    # 总金额时间加权
                    weighted_total_amount = np.sum(valid_on_px * valid_on_qty_remain * weights / 10000)

                    # 大单金额时间加权
                    weighted_large_order_amount = np.sum(
                        valid_on_px[large_order_mask] * valid_on_qty_remain[large_order_mask] * weights[large_order_mask] / 10000
                    )

                    # 计算大单占比
                    if weighted_total_amount > 0:
                        proportion = weighted_large_order_amount / weighted_total_amount
                    else:
                        proportion = 0

                    curr_dataset[index, col] = proportion
                else:
                    curr_dataset[index, col] = np.nan

            index += 1




