# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:30:24 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
from numba import njit, types, prange


# %%
@njit(types.int64[:](types.int64, types.int64[:]))
def get_residue_time(ts, ts_arr):
    """
    计算挂单剩余时间，考虑中午非交易时段，使用numba加速和向量化。
    
    参数:
    ts: int, 当前时间戳（13位毫秒级）
    ts_arr: numpy array, 挂单时间戳数组（13位毫秒级）
    
    返回:
    numpy array, 每个挂单的剩余时间（毫秒）
    """
    # 定义交易时段边界
    MORNING_END = 11 * 60 + 30  # 上午结束时间：11:30，单位分钟
    AFTERNOON_START = 13 * 60  # 下午开始时间：13:00，单位分钟
    NON_TRADING_INTERVAL = 90 * 60 * 1000  # 非交易时段90分钟，单位毫秒

    # 转换时间戳为分钟单位
    on_time_minutes = (ts_arr // 60000) % 1440
    current_time_minutes = (ts // 60000) % 1440

    # 计算初始时间差
    time_differences = ts - ts_arr

    # 标记无效数据（时间戳为0或时间差为负）
    is_valid = (ts_arr > 0) & (time_differences >= 0)

    # 初始化剩余时间数组
    residual_time = np.zeros_like(ts_arr, dtype=np.int64)

    # 对有效数据进行计算
    if np.any(is_valid):
        valid_differences = time_differences[is_valid]
        valid_on_time_minutes = on_time_minutes[is_valid]

        # 标记需要扣除非交易时间的条件
        needs_adjustment = (valid_on_time_minutes <= MORNING_END) & (current_time_minutes >= AFTERNOON_START)

        # 扣除非交易时间
        adjusted_differences = np.where(needs_adjustment, valid_differences - NON_TRADING_INTERVAL, valid_differences)

        # 确保剩余时间不为负
        residual_time[is_valid] = np.maximum(0, adjusted_differences)

    return residual_time


@njit(types.float64(types.float64, types.float64))
def safe_divide(a, b):
    try:
        return a / b
    except:
        return np.nan
   
    
@njit(types.float64[:](types.float64[:], types.float64[:]))
def safe_divide_arrays(arr1, arr2):
    # 初始化结果数组，类型为浮点数，以支持 np.nan
    result = np.empty_like(arr1, dtype=np.float64)
    
    for i in prange(len(arr1)):
        try:
            # 安全相除
            result[i] = arr1[i] / arr2[i]
        except:
            # 捕获除以零的情况
            result[i] = np.nan
    
    return result


@njit(types.float64[:](types.float64[:], types.float64))
def safe_divide_array_by_scalar(arr, scalar):
    # 初始化结果数组，类型为浮点数，以支持 np.nan
    result = np.empty_like(arr, dtype=np.float64)
    
    for i in prange(len(arr)):
        try:
            # 安全相除
            result[i] = arr[i] / scalar
        except:
            # 捕获除以零的情况
            result[i] = np.nan
    
    return result