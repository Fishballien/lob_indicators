# -*- coding: utf-8 -*-
"""
TRNetOPWeighted 因子：时间加权的近期挂单金额减去撤单金额

@author: Combined Logic

计算逻辑：加权挂单金额 - 加权撤单金额
- 挂单金额：on_qty_org * on_px / 10000
- 撤单金额：on_qty_d * on_px / 10000  
- 时间权重：(1 - normalized_time_diff)^alpha
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
    types.float64[:],    # alpha_values
    types.float64[:, :]  # curr_dataset
))
def TRNetOPW(best_px, on_side, on_px, on_qty_org, on_qty_d, on_ts_org, ts, 
             value_thresholds, time_ranges, alpha_values, curr_dataset):
    """
    计算时间加权的TRNetOP因子：加权挂单金额减去加权撤单金额
    
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
    - alpha_values: power衰减参数列表，权重 = (1-t)^alpha
    - curr_dataset: 存储结果数组，行对应参数组合，列对应 Bid 和 Ask
    
    计算逻辑：
    TRNetOPWeighted = 加权挂单金额 - 加权撤单金额
    
    其中：
    - 加权挂单金额 = Σ(on_qty_org * on_px / 10000 * weight)
    - 加权撤单金额 = Σ(on_qty_d * on_px / 10000 * weight)  
    - weight = (1 - normalized_time_diff)^alpha
    - normalized_time_diff = (ts - on_ts_org) / (time_range * 1000 * 60)
    
    结果存储顺序：
    - 外层循环：value_thresholds
    - 中层循环：time_ranges  
    - 内层循环：alpha_values
    - 每行存储：[bid_net_weighted_amount, ask_net_weighted_amount]
    """
    bid1 = best_px[0]
    ask1 = best_px[1]

    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return

    # 预计算所有挂单的金额 (向量化)
    amounts = on_px.astype(np.float64) * on_qty_org.astype(np.float64) / 10000.0
    
    # 创建买卖单掩码 (向量化)
    bid_mask = on_side == 0
    ask_mask = on_side == 1
    
    index = 0
    
    # 外层循环：金额阈值
    for T in value_thresholds:
        # 金额过滤掩码 (向量化)
        amount_mask = amounts >= T
        
        # 中层循环：时间范围
        for time_range in time_ranges:
            time_range_ms = time_range * 1000.0 * 60.0  # 转换为毫秒
            time_threshold = ts - time_range_ms
            
            # 时间过滤掩码 (向量化)
            time_mask = on_ts_org >= time_threshold
            
            # 计算归一化时间差 (向量化)
            time_diffs = (ts - on_ts_org.astype(np.float64)) / time_range_ms
            # 确保时间差在[0,1]范围内，超出范围的设为无效值
            valid_time_mask = (time_diffs >= 0.0) & (time_diffs <= 1.0)
            
            # 内层循环：alpha参数
            for alpha in alpha_values:
                
                # 计算power衰减权重 (向量化)
                weights = np.where(valid_time_mask, (1.0 - time_diffs) ** alpha, 0.0)
                
                # 组合所有过滤条件
                bid_final_mask = bid_mask & amount_mask & time_mask & valid_time_mask
                ask_final_mask = ask_mask & amount_mask & time_mask & valid_time_mask
                
                # === 计算加权挂单金额 ===
                # 计算挂单的加权金额 (向量化)
                order_weighted_amounts = amounts * weights
                
                # 计算Bid和Ask的加权挂单总金额
                bid_order_mask = bid_final_mask & (on_qty_org > 0)
                ask_order_mask = ask_final_mask & (on_qty_org > 0)
                
                bid_order_total = np.sum(order_weighted_amounts * bid_order_mask.astype(np.float64))
                ask_order_total = np.sum(order_weighted_amounts * ask_order_mask.astype(np.float64))
                
                # === 计算加权撤单金额 ===
                # 计算撤单的金额和权重
                cancel_amounts = on_px.astype(np.float64) * on_qty_d.astype(np.float64) / 10000.0
                cancel_weighted_amounts = cancel_amounts * weights
                
                # 计算Bid和Ask的加权撤单总金额
                bid_cancel_mask = bid_final_mask & (on_qty_d > 0)
                ask_cancel_mask = ask_final_mask & (on_qty_d > 0)
                
                bid_cancel_total = np.sum(cancel_weighted_amounts * bid_cancel_mask.astype(np.float64))
                ask_cancel_total = np.sum(cancel_weighted_amounts * ask_cancel_mask.astype(np.float64))
                
                # === 计算净额：挂单 - 撤单 ===
                bid_net_total = bid_order_total - bid_cancel_total
                ask_net_total = ask_order_total - ask_cancel_total
                
                # 存储结果
                curr_dataset[index, 0] = bid_net_total
                curr_dataset[index, 1] = ask_net_total
                
                index += 1


# 使用示例：
# value_thresholds = np.array([1000.0, 5000.0], dtype=np.float64)  # 2个金额阈值
# time_ranges = np.array([1.0, 5.0], dtype=np.float64)            # 2个时间范围(分钟)
# alpha_values = np.array([0.5, 1.0, 2.0], dtype=np.float64)      # 3个alpha参数
# 
# 结果数组：rows = 2*2*3 = 12, cols = 2 (bid和ask)
# curr_dataset = np.zeros((len(value_thresholds) * len(time_ranges) * len(alpha_values), 2), dtype=np.float64)
#
# 调用示例：
# TRNetOPWeighted(best_px, on_side, on_px, on_qty_org, on_qty_d, on_ts_org, ts,
#                 value_thresholds, time_ranges, alpha_values, curr_dataset)
#
# 功能说明：
# 1. 结合了Batch34的Order-Cancel净额逻辑
# 2. 应用了Batch28的时间衰减权重机制
# 3. 向量化运算提高性能
# 4. 分别计算挂单和撤单的加权金额，然后求差
#
# Alpha参数含义：
# - α = 0.5：平方根衰减，前期衰减快
# - α = 1.0：线性衰减  
# - α = 2.0：二次衰减，后期衰减快
# - α越大，近期权重保持越久，远期衰减越快
#
# 因子经济含义：
# - 正值：近期挂单力度大于撤单力度，市场流动性供给增加
# - 负值：近期撤单力度大于挂单力度，市场流动性供给减少
# - 时间权重确保近期行为影响更大，远期行为影响逐渐衰减