# -*- coding: utf-8 -*-
"""
TRNetOP 因子：近期挂单金额减去近期撤单金额（增强版）
新增 filter_minutes 参数，用于剔除部分开盘的单

@author: Enhanced from Batch34 with Batch31 filter functionality

计算逻辑：data_type1 - data_type3
- data_type1: 挂单金额（on_qty_org * on_px / 10000）
- data_type3: 撤单金额（on_qty_d * on_px / 10000）
"""

import numpy as np
from numba import njit, types


@njit(types.void(
    types.int64[:],      # best_px
    types.int32[:],      # on_side
    types.int64[:],      # on_px
    types.int64[:],      # on_qty_org
    types.int64[:],      # on_qty_d
    types.int64[:],      # on_ts_org
    types.int64,         # ts
    types.float64[:],    # value_thresholds
    types.float64[:],    # time_ranges
    types.float64[:],    # filter_minutes
    types.float64[:, :]  # curr_dataset
))
def TRNetOPRmAuc(best_px, on_side, on_px, on_qty_org, on_qty_d, on_ts_org, ts, 
                 value_thresholds, time_ranges, filter_minutes, curr_dataset):
    """
    计算TRNetOP因子：近期挂单金额减去近期撤单金额（带开盘时间过滤）
    
    参数：
    - best_px: 买一卖一价格 [bid1, ask1]
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 原始挂单量
    - on_qty_d: 撤单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为分钟
    - filter_minutes: 筛选分钟数列表，用于剔除开盘初期的单（如[25,30]表示分别计算9:25和9:30后的单）
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围*过滤时间），列对应 Bid 和 Ask
    
    计算逻辑：
    TRNetOP = 挂单金额 - 撤单金额
           = (on_qty_org * on_px / 10000) - (on_qty_d * on_px / 10000)
    
    筛选条件：
    - 方向匹配：on_side == side (0为买单，1为卖单)
    - 金额阈值：on_px * on_qty_org / 10000 >= T
    - 时间范围：on_ts_org >= time_threshold
    - 开盘过滤：on_ts_org >= filter_ts
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

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
                    # 基础条件：方向匹配、金额大于阈值、时间在范围内、晚于开盘过滤时间
                    base_mask = ((on_side == side) & 
                                (on_px * on_qty_org / 10000 >= T) & 
                                (on_ts_org >= time_threshold) & 
                                (on_ts_org >= filter_ts))
                    
                    # 计算挂单金额（data_type1）
                    order_amount = 0.0
                    mask_order = base_mask & (on_qty_org > 0)
                    if np.any(mask_order):
                        order_amount = np.sum(on_qty_org[mask_order] * on_px[mask_order] / 10000)
                    
                    # 计算撤单金额（data_type3）
                    cancel_amount = 0.0
                    mask_cancel = base_mask & (on_qty_d > 0)
                    if np.any(mask_cancel):
                        cancel_amount = np.sum(on_qty_d[mask_cancel] * on_px[mask_cancel] / 10000)
                    
                    # TRNetOP = 挂单金额 - 撤单金额
                    net_order_amount = order_amount - cancel_amount
                    
                    curr_dataset[index, col] = net_order_amount
                
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
TRNetOPWithFilter(
    best_px, on_side, on_px, on_qty_org, on_qty_d, on_ts_org, ts,
    value_thresholds, time_ranges, filter_minutes, curr_dataset
)
"""