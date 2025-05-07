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
    types.float64[:],  # boundary
    types.float64[:],  # weight_step
    types.float32[:, :]  # curr_dataset
))
def WeightedOrderValue(best_px, on_side, on_px, on_qty_remain, boundary, weight_step, curr_dataset):
    """
    计算加权挂单金额因子，Bid和Ask侧独立计算。
    - boundary: 百分比划分的界限
    - weight_step: 权重递减阶梯
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_px = (bid1 + ask1) / 2.0

    # 边界处理
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 初始化行索引
    index = 0

    # 遍历参数组合
    for b in boundary:
        for w in weight_step:
            # Bid侧计算
            bid_idx = (on_side == 0)
            if np.any(bid_idx):
                bid_weights = np.where(
                    np.abs(on_px[bid_idx] - mid_px) / mid_px <= b,
                    1 - w,
                    1
                )
                curr_dataset[index, 0] = np.sum(bid_weights * on_qty_remain[bid_idx] * (on_px[bid_idx] / 10000))
            else:
                curr_dataset[index, 0] = 0

            # Ask侧计算
            ask_idx = (on_side == 1)
            if np.any(ask_idx):
                ask_weights = np.where(
                    np.abs(on_px[ask_idx] - mid_px) / mid_px <= b,
                    1 - w,
                    1
                )
                curr_dataset[index, 1] = np.sum(ask_weights * on_qty_remain[ask_idx] * (on_px[ask_idx] / 10000))
            else:
                curr_dataset[index, 1] = 0

            # 更新行索引
            index += 1
            
            
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:],  # boundary
    types.float64[:],  # near_weight
    types.float64[:],  # far_weight
    types.float32[:, :]  # curr_dataset
))
def AdjustableSegmentedLinearWeightedOrderValue(best_px, on_side, on_px, on_qty_remain, boundary, near_weight, far_weight, curr_dataset):
    bid1 = best_px[0]
    ask1 = best_px[1]
    mid_px = (bid1 + ask1) / 2.0

    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0

    for b in boundary:
        for n_w in near_weight:
            for f_w in far_weight:
                # Bid侧计算
                bid_idx = (on_side == 0)
                if np.any(bid_idx):
                    bid_dist = np.abs(on_px[bid_idx] - mid_px) / mid_px
                    bid_weights = np.where(
                        bid_dist <= b,
                        n_w + (1 - n_w) * (1 - bid_dist / b),
                        f_w + (1 - f_w) * (bid_dist - b) / (1 - b)
                    )
                    curr_dataset[index, 0] = np.sum(bid_weights * on_qty_remain[bid_idx] * (on_px[bid_idx] / 10000))
                else:
                    curr_dataset[index, 0] = 0

                # Ask侧计算
                ask_idx = (on_side == 1)
                if np.any(ask_idx):
                    ask_dist = np.abs(on_px[ask_idx] - mid_px) / mid_px
                    ask_weights = np.where(
                        ask_dist <= b,
                        n_w + (1 - n_w) * (1 - ask_dist / b),
                        f_w + (1 - f_w) * (ask_dist - b) / (1 - b)
                    )
                    curr_dataset[index, 1] = np.sum(ask_weights * on_qty_remain[ask_idx] * (on_px[ask_idx] / 10000))
                else:
                    curr_dataset[index, 1] = 0
                index += 1

