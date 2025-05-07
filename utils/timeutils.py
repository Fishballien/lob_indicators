# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:02:48 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


# %%
def get_a_share_intraday_time_series(date: datetime, params):
    """
    生成A股市场交易时间内的等间隔时间序列。
    
    :param date: 日期 (datetime对象)
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1} 或 {'minutes': 1})
    :return: numpy数组，包含当天交易时间内的时间戳序列 (毫秒级)
    """
    # 定义A股市场交易时段
    morning_start = datetime(date.year, date.month, date.day, 9, 30)
    morning_end = datetime(date.year, date.month, date.day, 11, 30)
    afternoon_start = datetime(date.year, date.month, date.day, 13, 0)
    afternoon_end = datetime(date.year, date.month, date.day, 15, 0)

    interval = timedelta(**params)
    
    # 生成上午交易时间序列
    morning_series = np.arange(morning_start + interval, morning_end + interval, 
                               interval).astype('i8') // 1e3
    
    # 生成下午交易时间序列
    afternoon_series = np.arange(afternoon_start + interval, afternoon_end + interval, 
                                 interval).astype('i8') // 1e3
    
    # 合并上午和下午时间序列
    time_series = np.concatenate([morning_series, afternoon_series]).astype(np.int64)
    
    return time_series


# =============================================================================
# def get_a_share_intraday_time_series(date: datetime, params):
#     """
#     生成A股市场交易时间内的等间隔时间序列（北京时间）。
#     
#     :param date: 日期 (datetime对象)
#     :param params: 时间间隔参数 (字典形式，如 {'seconds': 1} 或 {'minutes': 1})
#     :return: numpy数组，包含当天交易时间内的时间戳序列 (毫秒级)
#     """
#     # 设置北京时间时区
#     beijing_tz = timezone(timedelta(hours=8))
# 
#     interval = timedelta(**params)
#     
#     # 定义A股市场交易时段（以北京时间为基准）
#     morning_start = datetime(date.year, date.month, date.day, 9, 30, tzinfo=beijing_tz) + interval
#     morning_end = datetime(date.year, date.month, date.day, 11, 30, tzinfo=beijing_tz) + interval
#     afternoon_start = datetime(date.year, date.month, date.day, 13, 0, tzinfo=beijing_tz) + interval
#     afternoon_end = datetime(date.year, date.month, date.day, 15, 0, tzinfo=beijing_tz) + interval
#     
#     # 生成上午交易时间序列
#     morning_series = np.arange(morning_start.timestamp() * 1e3, morning_end.timestamp() * 1e3, 
#                                interval.total_seconds() * 1e3).astype('i8')
#     
#     # 生成下午交易时间序列
#     afternoon_series = np.arange(afternoon_start.timestamp() * 1e3, afternoon_end.timestamp() * 1e3, 
#                                  interval.total_seconds() * 1e3).astype('i8')
#     
#     # 合并上午和下午时间序列
#     time_series = np.concatenate([morning_series, afternoon_series])
#     
#     return time_series
# =============================================================================


def generate_date_range(start_date_str, end_date_str):
    date_range = pd.date_range(start=start_date_str, end=end_date_str)
    return date_range.strftime("%Y%m%d").tolist()


def adjust_timestamp_precision(time_arr):
    # 确保 time_arr 是整数数组
    time_arr = time_arr.astype('int64')  # 转换为 64 位整数以避免精度损失
    # 检查第一个时间戳的位数
    first_timestamp = str(time_arr[0])
    length = len(first_timestamp)
    
    if length == 16:
        adjusted_time_arr = (time_arr // 1_000).astype('i8')
    elif length == 19:
        adjusted_time_arr = (time_arr // 1_000_000).astype('i8')
    else:
        # 如果时间戳已经是秒级 (10位)，无需调整
        adjusted_time_arr = time_arr.astype('i8')
    
    return adjusted_time_arr


def generate_time_series_in_date_range(start_date: datetime, end_date: datetime, params: dict):
    """
    给定开始日期和结束日期，生成指定时间间隔的所有交易时间段时间序列。

    :param start_date: 开始日期 (datetime对象)
    :param end_date: 结束日期 (datetime对象)
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1} 或 {'minutes': 1})
    :return: numpy.datetime64 数组，包含所有日期交易时间段内的时间戳序列
    """
    # 预生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # 生成每天的时间序列
    all_series = [get_a_share_intraday_time_series(date, params) for date in date_range]

    # 合并所有日期的时间序列
    full_series = np.concatenate(all_series)

    return full_series.view('datetime64[ms]')