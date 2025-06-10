# -*- coding: utf-8 -*-
"""
Created on [Current Date]

@author: [Your Name]

计算剩余挂单的金额加权距离因子
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
    types.int64[:],  # on_qty_org
    types.int64[:],  # on_ts_org
    types.int64,     # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # time_ranges
    types.float64[:, :]  # curr_dataset
))
def AmtWDist(best_px, on_side, on_px, on_qty_remain, on_qty_org, on_ts_org, ts, value_thresholds, time_ranges, curr_dataset):
    """
    计算不同挂单金额、不同时间范围内的挂单金额加权距离。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_remain: 当前剩余挂单量
    - on_qty_org: 原始挂单量
    - on_ts_org: 挂单时间戳（13位毫秒）
    - ts: 当前时间戳
    - value_thresholds: 金额阈值列表，单位为原始货币
    - time_ranges: 时间范围列表，单位为毫秒
    - curr_dataset: 存储结果数组，行对应参数组合（金额阈值*时间范围），列对应 Bid 和 Ask
    
    计算逻辑：
    - 金额筛选使用 on_px * on_qty_org / 10000 >= T 进行判断
    - 对于买单：距离百分比 = (mid - bid_price) / mid
    - 对于卖单：距离百分比 = (ask_price - mid) / mid
    - 最终因子值 = Σ(距离百分比 * 挂单金额)
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    
    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 计算中间价
    mid_price = (bid1 + ask1) / 2.0
    
    index = 0
    for T in value_thresholds:  # 遍历所有金额阈值
        for time_range in time_ranges:  # 遍历所有时间范围
            time_threshold = ts - time_range * 1000 * 60  # 计算时间阈值
            
            # Bid 和 Ask 侧分别处理
            for side, col in [(0, 0), (1, 1)]:
                # 使用原始挂单量(on_qty_org)进行金额筛选
                mask = (on_side == side) & (on_px * on_qty_org / 10000 >= T) & (on_ts_org >= time_threshold)
                
                if np.any(mask):
                    # 获取符合条件的挂单数据
                    filtered_px = on_px[mask]
                    filtered_qty_remain = on_qty_remain[mask]
                    
                    # 计算每笔挂单的金额（使用剩余量）
                    amounts = filtered_px * filtered_qty_remain / 10000.0
                    
                    # 计算距离百分比
                    if side == 0:  # 买单
                        # 距离百分比 = (mid - bid_price) / mid
                        distance_ratios = (mid_price - filtered_px) / mid_price
                    else:  # 卖单
                        # 距离百分比 = (ask_price - mid) / mid
                        distance_ratios = (filtered_px - mid_price) / mid_price
                    
                    # 计算金额加权距离（直接计算加权和，不除以总金额）
                    weighted_distance_sum = np.sum(distance_ratios * amounts)
                    curr_dataset[index, col] = weighted_distance_sum
                else:
                    curr_dataset[index, col] = np.nan  # 无符合条件的挂单记为NaN
            
            index += 1