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
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # price_ranges
    types.float32[:, :]  # curr_dataset
))
def NearOrderAmountRatio(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, price_ranges, curr_dataset):
    """
    计算近处挂单金额占比 (Near Order Amount Ratio)
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - price_ranges: 价格范围列表，每个范围为相对于中间价的百分比
    - curr_dataset: 存储结果数组，行对应不同 price_range 值，列对应 Bid 和 Ask
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2  # 中间价

    # 遍历每个价格范围
    for idx, price_range in enumerate(price_ranges):
        lower_bound = mid_price * (1 - price_range)  # 计算价格范围的下界
        upper_bound = mid_price * (1 + price_range)  # 计算价格范围的上界

        # Bid 和 Ask 侧分别处理
        for side, col in [(0, 0), (1, 1)]:  # Bid: 0列, Ask: 1列
            mask = on_side == side
            if np.any(mask):
                valid_on_px = on_px[mask]
                valid_on_qty_remain = on_qty_remain[mask]

                # 筛选在价格范围内的挂单
                in_range = (valid_on_px >= lower_bound) & (valid_on_px <= upper_bound)

                # 计算在该范围内的挂单金额
                near_order_amount = np.sum(valid_on_px[in_range] * valid_on_qty_remain[in_range])

                # 计算所有挂单的金额
                total_order_amount = np.sum(valid_on_px * valid_on_qty_remain)

                # 计算近处挂单金额占比
                if total_order_amount > 0:
                    curr_dataset[idx, col] = near_order_amount / total_order_amount
                else:
                    curr_dataset[idx, col] = np.nan
            else:
                curr_dataset[idx, col] = np.nan


@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # price_ranges
    types.float32[:, :]  # curr_dataset
))
def FarOrderAmountProportion(best_px, on_side, on_px, on_qty_remain, on_ts_org, ts, price_ranges, curr_dataset):
    bid1 = best_px[0]
    ask1 = best_px[1]

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    mid_price = (bid1 + ask1) / 2  # 中间价

    for idx, price_range in enumerate(price_ranges):
        lower_bound_bid = bid1 * (1 - price_range)
        upper_bound_bid = bid1 * (1 + price_range)

        lower_bound_ask = ask1 * (1 - price_range)
        upper_bound_ask = ask1 * (1 + price_range)

        for side, col in [(0, 0), (1, 1)]:
            mask = on_side == side
            if np.any(mask):
                valid_on_px = on_px[mask]
                valid_on_qty_remain = on_qty_remain[mask]

                # 计算远离中间价的条件
                if side == 0:  # Bid 侧
                    in_far_range = (valid_on_px < lower_bound_bid) | (valid_on_px > upper_bound_bid)
                else:  # Ask 侧
                    in_far_range = (valid_on_px < lower_bound_ask) | (valid_on_px > upper_bound_ask)

                # 远处挂单金额：价格 * 剩余数量
                far_order_amount = np.sum(valid_on_px[in_far_range] * valid_on_qty_remain[in_far_range])
                total_order_amount = np.sum(valid_on_px * valid_on_qty_remain)

                if total_order_amount > 0:
                    curr_dataset[idx, col] = far_order_amount / total_order_amount
                else:
                    curr_dataset[idx, col] = np.nan
            else:
                curr_dataset[idx, col] = np.nan
