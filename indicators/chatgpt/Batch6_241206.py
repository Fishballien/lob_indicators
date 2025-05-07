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
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:, :]  # curr_dataset
))
def PriceTimeWeightedShift(best_px, on_side, on_px, on_ts_org, ts, curr_dataset):
    """
    计算挂单价格时间权重偏移因子 (Price-Time Weighted Shift)
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - curr_dataset: 存储结果数组，第一列为 Bid，第二列为 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid 和 Ask 侧分别处理
    for side, col in [(0, 0), (1, 1)]:  # Bid: 0列, Ask: 1列
        mask = on_side == side
        if np.any(mask):
            valid_on_px = on_px[mask]
            valid_on_ts_org = on_ts_org[mask]

            time_weights = ts - valid_on_ts_org
            if np.sum(time_weights) == 0:
                curr_dataset[:, col] = np.nan
            else:
                price_weighted_sum = np.sum(valid_on_px * time_weights)
                time_weight_sum = np.sum(time_weights)
                curr_dataset[:, col] = price_weighted_sum / time_weight_sum
        else:
            curr_dataset[:, col] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:, :]  # curr_dataset
))
def AmountTimeConcentration(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, curr_dataset):
    """
    计算挂单金额时间密集度因子 (Amount-Time Concentration)
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - curr_dataset: 存储结果数组，第一列为 Bid，第二列为 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid 和 Ask 侧分别处理
    for side, col in [(0, 0), (1, 1)]:  # Bid: 0列, Ask: 1列
        mask = on_side == side
        if np.any(mask):
            valid_on_px = on_px[mask]
            valid_on_qty_remain = on_qty_remain[mask]
            valid_on_ts_org = on_ts_org[mask]

            # 时间权重计算：1 / (ts - on_ts_org + 1)
            time_weights = 1 / (ts - valid_on_ts_org + 1)
            # 挂单金额加权求和
            weighted_sum = np.sum(valid_on_px * valid_on_qty_remain * time_weights)

            curr_dataset[:, col] = weighted_sum
        else:
            curr_dataset[:, col] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # lambda_values
    types.float64[:, :]  # curr_dataset
))
def DecayWeightedAmount(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, lambda_values, curr_dataset):
    """
    计算挂单金额衰减因子 (Decay Weighted Amount)
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - lambda_values: 衰减系数数组
    - curr_dataset: 存储结果数组，第一列为 Bid，第二列为 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 遍历所有 lambda 值
    index = 0
    for lambd in lambda_values:
        # Bid 和 Ask 侧分别处理
        for side, col in [(0, 0), (1, 1)]:  # Bid: 0列, Ask: 1列
            mask = on_side == side
            if np.any(mask):
                valid_on_px = on_px[mask]
                valid_on_qty_remain = on_qty_remain[mask]
                valid_on_ts_org = on_ts_org[mask]

                # 计算衰减权重：e^{-λ (ts - on_ts_org)}
                time_diff = ts - valid_on_ts_org
                decay_weights = np.exp(-lambd * time_diff)

                # 计算加权金额和
                decay_weighted_sum = np.sum(valid_on_px * valid_on_qty_remain * decay_weights)
                curr_dataset[index, col] = decay_weighted_sum
            else:
                curr_dataset[index, col] = np.nan

        # 更新行索引
        index += 1


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # decay_list
    types.float32[:, :]  # curr_dataset
))
def TimeBucketedOrderAmount(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, decay_list, curr_dataset):
    """
    计算挂单时间分层金额因子 (Time-Bucketed Order Amount)
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - decay_list: 衰减值列表，逐一遍历
    - curr_dataset: 存储结果数组，行对应不同 decay 值，列对应 Bid 和 Ask
    """
    # 时间区间边界（单位毫秒）
    time_buckets = [10 * 1000, 60 * 1000, 10 * 60 * 1000, 30 * 60 * 1000]
    num_buckets = len(time_buckets) + 1  # 包括更高时间层级

    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 遍历每个 decay 值
    for idx, decay in enumerate(decay_list):
        # 生成每层的权重，最近的层权重为1，依次递减
        weights = 1 - decay * np.arange(num_buckets)
        weights = np.maximum(weights, 0)  # 确保权重不为负
        if np.sum(weights) > 0:
            weights /= np.sum(weights)  # 归一化权重

        # Bid 和 Ask 侧分别处理
        for side, col in [(0, 0), (1, 1)]:  # Bid: 0列, Ask: 1列
            mask = on_side == side
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
                    bucket_amounts[i] = np.sum(valid_on_px[in_bucket] * valid_on_qty_remain[in_bucket])
                    time_deltas[in_bucket] = np.inf  # 防止多次分配

                # 剩余部分为最高时间段
                bucket_amounts[-1] = np.sum(valid_on_px[time_deltas != np.inf] * valid_on_qty_remain[time_deltas != np.inf])

                # 计算加权平均
                weighted_sum = np.sum(bucket_amounts * weights)
                curr_dataset[idx, col] = weighted_sum
            else:
                curr_dataset[idx, col] = np.nan
