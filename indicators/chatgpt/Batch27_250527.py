# -*- coding: utf-8 -*-
"""
Created on Mon May 25 2025

订单生命周期效率因子 - 识别智能资金的核心指标
"""
# %% imports
import numpy as np
from numba import njit, types


# %%
@njit(types.void(
    types.int64[:],    # best_px
    types.int32[:],    # on_side  
    types.int64[:],    # on_px
    types.int64[:],    # on_qty_org
    types.int64[:],    # on_ts_org
    types.int64[:],    # on_qty_d
    types.int64[:],    # on_ts_d
    types.int64[:],    # on_qty_t
    types.int64[:],    # on_ts_t
    types.int64[:],    # on_amt_t
    types.int64,       # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # time_ranges
    types.float64[:, :] # curr_dataset
))
def OrderLifecycleEfficiency(best_px, on_side, on_px, on_qty_org, on_ts_org, on_qty_d, on_ts_d,
                            on_qty_t, on_ts_t, on_amt_t, ts, value_thresholds, time_ranges, curr_dataset):
    """
    计算订单生命周期效率因子，识别聪明钱特征
    
    核心逻辑：
    1. 执行效率 = 成交金额 / 原始挂单金额 (衡量订单完成度)
    2. 时间效率 = 1 / (平均成交时间 - 挂单时间) (衡量执行速度)
    3. 撤单惩罚 = 撤单金额 / 原始挂单金额 (衡量决策质量)
    
    综合得分 = 执行效率 * 时间效率 * (1 - 撤单惩罚)
    高分表示快速高质量的订单执行，是聪明钱的典型特征
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    index = 0
    for T in value_thresholds:
        for time_range in time_ranges:
            time_threshold = ts - time_range * 1000 * 60
            
            for side, col in [(0, 0), (1, 1)]:
                # 筛选大额订单
                large_order_mask = (on_side == side) & (on_qty_org * on_px / 10000 >= T) & (on_ts_org >= time_threshold) & (on_ts_org > 0)
                
                if not np.any(large_order_mask):
                    curr_dataset[index, col] = 0.0
                    continue
                
                total_score = 0.0
                valid_orders = 0
                
                # 遍历每个符合条件的大额订单
                for i in range(len(on_side)):
                    if not large_order_mask[i]:
                        continue
                        
                    original_amount = on_px[i] * on_qty_org[i] / 10000.0
                    if original_amount == 0:
                        continue
                    
                    # 1. 执行效率 (0-1)
                    if on_qty_t[i] > 0 and on_amt_t[i] > 0:
                        execution_efficiency = (on_amt_t[i] / 10000.0) / original_amount
                        execution_efficiency = min(execution_efficiency, 1.0)  # 防止超过1
                    else:
                        execution_efficiency = 0.0
                    
                    # 2. 时间效率 (基于成交速度)
                    time_efficiency = 0.0
                    if on_qty_t[i] > 0 and on_ts_t[i] > on_ts_org[i]:
                        execution_time_minutes = (on_ts_t[i] - on_ts_org[i]) / 60000.0  # 转换为分钟
                        if execution_time_minutes > 0:
                            # 成交越快得分越高，使用指数衰减
                            time_efficiency = np.exp(-execution_time_minutes / 5.0)  # 5分钟半衰期
                    
                    # 3. 撤单惩罚 (0-1，越高越差)
                    cancel_penalty = 0.0
                    if on_qty_d[i] > 0:
                        cancel_amount = on_px[i] * on_qty_d[i] / 10000.0
                        cancel_penalty = cancel_amount / original_amount
                        cancel_penalty = min(cancel_penalty, 1.0)
                    
                    # 4. 综合评分
                    if execution_efficiency > 0:  # 只计算有成交的订单
                        order_score = execution_efficiency * time_efficiency * (1.0 - cancel_penalty)
                        total_score += order_score * original_amount  # 按金额加权
                        valid_orders += 1
                
                # 计算加权平均效率
                if valid_orders > 0:
                    # 按订单数量标准化，避免单纯的金额驱动
                    curr_dataset[index, col] = total_score / valid_orders
                else:
                    curr_dataset[index, col] = 0.0
            
            index += 1


@njit(types.void(
    types.int64[:],    # best_px
    types.int32[:],    # on_side
    types.int64[:],    # on_px  
    types.int64[:],    # on_qty_org
    types.int64[:],    # on_ts_org
    types.int64[:],    # on_qty_d
    types.int64[:],    # on_ts_d
    types.int64,       # ts
    types.float64[:],  # value_thresholds
    types.float64[:],  # time_ranges
    types.float64[:, :] # curr_dataset
))
def OrderPersistenceRatio(best_px, on_side, on_px, on_qty_org, on_ts_org, on_qty_d, on_ts_d,
                         ts, value_thresholds, time_ranges, curr_dataset):
    """
    计算订单坚持度比率 - 衡量订单挂出后的坚持程度
    
    坚持度 = 未撤单订单数量 / 总订单数量
    高值表示强烈的交易意图，低值表示试探性挂单
    """
    bid1 = best_px[0]
    ask1 = best_px[1]
    
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    index = 0
    for T in value_thresholds:
        for time_range in time_ranges:
            time_threshold = ts - time_range * 1000 * 60
            
            for side, col in [(0, 0), (1, 1)]:
                # 筛选大额订单
                large_order_mask = (on_side == side) & (on_qty_org * on_px / 10000 >= T) & (on_ts_org >= time_threshold) & (on_ts_org > 0)
                
                if not np.any(large_order_mask):
                    curr_dataset[index, col] = np.nan
                    continue
                
                total_orders = np.sum(large_order_mask)
                persistent_orders = 0
                
                # 计算未撤单的订单数量
                for i in range(len(on_side)):
                    if large_order_mask[i]:
                        # 订单仍有剩余量且未被完全撤销
                        if on_qty_d[i] == 0 or on_ts_d[i] == 0:  # 从未撤单
                            persistent_orders += 1
                        elif on_qty_d[i] < on_qty_org[i]:  # 部分撤单但仍有剩余
                            persistent_orders += 1
                
                # 计算坚持度比率
                if total_orders > 0:
                    curr_dataset[index, col] = persistent_orders / total_orders
                else:
                    curr_dataset[index, col] = np.nan
            
            index += 1