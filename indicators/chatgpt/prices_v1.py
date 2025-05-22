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
    types.float64[:, :]  # curr_dataset
))
def BidAskPrice(best_px, curr_dataset):
    """
    简化版函数：仅将bid1和ask1价格直接写入curr_dataset。
    
    参数:
    best_px: 包含最优买卖价格的数组，索引0为bid1，索引1为ask1
    curr_dataset: 输出数据集，第一列为bid1，第二列为ask1
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    
    # 边界处理：如果买1或卖1价格无效，直接填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 将bid1和ask1写入curr_dataset的第一行
    # 对所有行都填充相同的值
    for i in range(curr_dataset.shape[0]):
        curr_dataset[i, 0] = bid1  # 第一列为bid1
        curr_dataset[i, 1] = ask1  # 第二列为ask1