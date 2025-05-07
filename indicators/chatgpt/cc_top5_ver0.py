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


# %%
@njit(types.void(
    types.int64[:],  # best_px
    types.int32[:],  # on_side
    types.int64[:],  # on_px
    types.int64[:],  # on_qty_remain
    types.float64[:, :]  # curr_dataset
))
def Top5PriceLevelVolumeSorted(best_px, on_side, on_px, on_qty_remain, curr_dataset):
    """
    计算排序后前五档不同价格上的挂单金额总量。
    限制条件：
      - Bid侧价格必须 <= Bid1
      - Ask侧价格必须 >= Ask1
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # Bid侧处理
    bid_prices = on_px[(on_side == 0) & (on_px <= bid1)]  # 筛选 <= Bid1 的价格
    bid_quantities = on_qty_remain[(on_side == 0) & (on_px <= bid1)]
    if len(bid_prices) > 0:
        # 按价格从大到小排序，并取前五个不同价格
        unique_bid_prices = np.unique(-np.sort(-bid_prices))[:5]  # 取前5档价格（从高到低）
        bid_volume = 0
        for price in unique_bid_prices:
            bid_volume += np.sum(bid_quantities[bid_prices == price] * price / 10000)
        curr_dataset[0, 0] = bid_volume
    else:
        curr_dataset[0, 0] = 0

    # Ask侧处理
    ask_prices = on_px[(on_side == 1) & (on_px >= ask1)]  # 筛选 >= Ask1 的价格
    ask_quantities = on_qty_remain[(on_side == 1) & (on_px >= ask1)]
    if len(ask_prices) > 0:
        # 按价格从小到大排序，并取前五个不同价格
        unique_ask_prices = np.unique(np.sort(ask_prices))[:5]  # 取前5档价格（从低到高）
        ask_volume = 0
        for price in unique_ask_prices:
            ask_volume += np.sum(ask_quantities[ask_prices == price] * price / 10000)
        curr_dataset[0, 1] = ask_volume
    else:
        curr_dataset[0, 1] = 0
