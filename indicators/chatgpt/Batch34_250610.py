# -*- coding: utf-8 -*-
"""
TRNetOP 因子：近期挂单金额减去近期撤单金额

@author: Claude

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
    types.float64[:, :]  # curr_dataset
))
def TRNetOP(best_px, on_side, on_px, on_qty_org, on_qty_d, on_ts_org, ts, 
            value_thresholds, time_ranges, curr_dataset):
    """
    计算TRNetOP因子：近期挂单金额减去近期撤单金额
    
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
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围），列对应 Bid 和 Ask
    
    计算逻辑：
    TRNetOP = 挂单金额 - 撤单金额
           = (on_qty_org * on_px / 10000) - (on_qty_d * on_px / 10000)
    
    筛选条件：
    - 方向匹配：on_side == side (0为买单，1为卖单)
    - 金额阈值：on_px * on_qty_org / 10000 >= T
    - 时间范围：on_ts_org >= time_threshold
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for time_range in time_ranges:  # 遍历所有时间范围
            time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值（转换为毫秒）
            
            # Bid 和 Ask 侧分别处理
            for side, col in [(0, 0), (1, 1)]:
                # 基础条件：方向匹配、金额大于阈值、时间在范围内
                base_mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= time_threshold)
                
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