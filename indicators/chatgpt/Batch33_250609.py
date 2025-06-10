# -*- coding: utf-8 -*-
"""
结合 Batch18 和 Batch21 的近期大单挂单因子计算函数
新增 filter_minutes 参数，用于剔除部分开盘的单
修改：使用中间价计算挂单金额，特别是对于ask端的挂单

@author: Combined from Batch18 and Batch21, Modified for mid-price calculation
"""

import numpy as np
from numba import njit, types


@njit(types.void(
    types.int64[:],      # best_px
    types.int32[:],      # on_side
    types.int64[:],      # on_px
    types.int64[:],      # on_qty_org
    types.int64[:],      # on_ts_org
    types.int64,         # ts
    types.float64[:],    # value_thresholds
    types.float64[:],    # time_ranges
    types.float64[:],    # filter_minutes
    types.float64[:, :]  # curr_dataset
))
def TROAMpc(best_px, on_side, on_px, on_qty_org, on_ts_org, ts, 
            value_thresholds, time_ranges, filter_minutes, curr_dataset):
    """
    计算近期大单挂单因子，结合时间范围筛选和开盘时间过滤。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 初始挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为分钟
    - filter_minutes: 筛选分钟数列表，用于剔除开盘初期的单（如[25,30]表示分别计算9:25和9:30后的单）
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围*过滤时间），列对应 Bid 和 Ask
    
    修改：
    - 对于 bid side (side=0)：保持原来的计算方式，使用 on_px * on_qty_org / 10000
    - 对于 ask side (side=1)：使用中间价计算，mid_price * on_qty_org / 10000
    - 函数会遍历所有金额阈值、时间范围和过滤时间的组合
    - 只统计在指定时间范围内且晚于开盘过滤时间的挂单
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 计算中间价
    mid_price = (bid1 + ask1) / 2

    # 计算当天0点的时间戳
    day_start_ts = (ts // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)

    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for time_range in time_ranges:  # 遍历所有时间范围
            for filter_minute in filter_minutes:  # 遍历所有过滤时间
                # 计算时间阈值（向前推算的时间范围）
                time_threshold = ts - time_range * 1000 * 60  # 转换分钟为毫秒
                
                # 转换filter_minute为int64类型
                filter_minute_int = np.int64(filter_minute)
                
                # 计算9:filter_minute的时间戳（开盘过滤时间）
                filter_ts = day_start_ts + (9 * 60 + filter_minute_int) * 60 * 1000
                
                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    if side == 0:  # Bid side - 保持原来的计算方式
                        # 符合条件的挂单：
                        # 1. 方向匹配
                        # 2. 金额大于阈值（使用挂单价格判断）
                        # 3. 时间在指定范围内（近期）
                        # 4. 晚于开盘过滤时间
                        mask = ((on_side == side) & 
                               (on_px * on_qty_org / 10000 >= T) & 
                               (on_ts_org >= time_threshold) & 
                               (on_ts_org >= filter_ts))
                        
                        if np.any(mask):
                            # 计算符合条件的挂单总金额，使用挂单价格计算
                            total_amount = np.sum(on_px[mask] * on_qty_org[mask] / 10000)
                            curr_dataset[index, col] = total_amount
                        else:
                            curr_dataset[index, col] = 0  # 无符合条件的挂单记为0
                    
                    else:  # Ask side - 使用中间价计算
                        # 符合条件的挂单：
                        # 1. 方向匹配
                        # 2. 金额大于阈值（使用中间价判断）
                        # 3. 时间在指定范围内（近期）
                        # 4. 晚于开盘过滤时间
                        mask = ((on_side == side) & 
                               (mid_price * on_qty_org / 10000 >= T) & 
                               (on_ts_org >= time_threshold) & 
                               (on_ts_org >= filter_ts))
                        
                        if np.any(mask):
                            # 计算符合条件的挂单总金额，使用中间价计算
                            total_amount = np.sum(mid_price * on_qty_org[mask] / 10000)
                            curr_dataset[index, col] = total_amount
                        else:
                            curr_dataset[index, col] = 0  # 无符合条件的挂单记为0
                
                index += 1


# 使用示例：
"""
# 金额阈值列表（例如：1000, 5000, 10000 元）
value_thresholds = np.array([1000.0, 5000.0, 10000.0], dtype=np.float64)

# 时间范围列表（例如：1分钟、5分钟、30分钟）
time_ranges = np.array([1.0, 5.0, 30.0], dtype=np.float64)

# 过滤时间列表（例如：9:25和9:30后的单）
filter_minutes = np.array([25.0, 30.0], dtype=np.float64)

# 结果数组：rows = len(value_thresholds) * len(time_ranges) * len(filter_minutes), cols = 2 (bid和ask)
curr_dataset = np.zeros((len(value_thresholds) * len(time_ranges) * len(filter_minutes), 2), dtype=np.float64)

# 调用函数
TROARmAuc(
    best_px, on_side, on_px, on_qty_org, on_ts_org, ts,
    value_thresholds, time_ranges, filter_minutes, curr_dataset
)
"""