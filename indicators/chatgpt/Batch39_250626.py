# -*- coding: utf-8 -*-
"""
基于 Batch22 修改的 ValueTimeDecayOrgOA 函数
修改内容：
1. time_buckets 以外的更长时间不赋予权重，只叠加 time_buckets 内的全部量
2. 加入 filter_minutes 参数，过滤集合竞价时间

@author: Modified from Batch22
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
    types.float64[:],    # decay_list
    types.float64[:],    # filter_minutes
    types.float64[:, :]  # curr_dataset
))
def ValueTimeDecayOrgOARec(best_px, on_side, on_px, on_qty_org, on_ts_org, ts, 
                           value_thresholds, decay_list, filter_minutes, curr_dataset):
    """
    计算筛选初始挂单量并对挂单时间做衰减的原始挂单金额因子（修改版）。
    
    参数：
    - best_px: 买一卖一价格
    - on_side: 挂单方向（0: 买单, 1: 卖单）
    - on_px: 挂单价格
    - on_qty_org: 初始挂单量
    - on_ts_org: 挂单时间戳
    - ts: 当前时间戳
    - value_thresholds: 大单金额筛选阈值
    - decay_list: 时间衰减参数列表
    - filter_minutes: 筛选分钟数列表，用于剔除开盘初期的单（如[25,30]表示分别计算9:25和9:30后的单）
    - curr_dataset: 存储结果数组，行对应不同参数组合，列对应 Bid 和 Ask
    
    修改说明：
    1. time_buckets 以外的更长时间不赋予权重，只统计 time_buckets 内的量
    2. 加入开盘时间过滤，剔除集合竞价阶段的挂单
    
    注：
    - 金额计算采用 on_px * on_qty_org / 10000
    - 时间衰减权重基于预定义的时间桶，但超出最大时间桶的订单不参与计算
    """
    # 定义时间分桶（毫秒）
    time_buckets = [10 * 1000, 60 * 1000, 5 * 60 * 1000, 10 * 60 * 1000, 15 * 60 * 1000, 30 * 60 * 1000]
    num_buckets = len(time_buckets)  # 不再包含最后一个桶（超长时间）
    
    bid1 = best_px[0]
    ask1 = best_px[1]
    
    # 边界处理：如果买一或卖一价格无效，填充 NaN
    if bid1 == 0 or ask1 == 0:
        curr_dataset[:, :] = np.nan
        return
    
    # 计算当天0点的时间戳
    day_start_ts = (ts // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)
    
    index = 0
    for T in value_thresholds:
        for decay in decay_list:
            for filter_minute in filter_minutes:
                # 生成每层的权重（只对time_buckets内的时间段）
                weights = 1 - decay * np.arange(num_buckets)
                weights = np.maximum(weights, 0)  # 确保权重非负
                
                if np.sum(weights) > 0:
                    weights /= np.sum(weights)  # 归一化权重
                
                # 转换filter_minute为int64类型
                filter_minute_int = np.int64(filter_minute)
                
                # 计算9:filter_minute的时间戳（开盘过滤时间）
                filter_ts = day_start_ts + (9 * 60 + filter_minute_int) * 60 * 1000
                
                # Bid 和 Ask 侧分别处理
                for side, col in [(0, 0), (1, 1)]:
                    # 筛选符合条件的订单：
                    # 1. 方向匹配
                    # 2. 金额大于阈值  
                    # 3. 晚于开盘过滤时间
                    mask = ((on_side == side) & 
                           (on_px * on_qty_org / 10000 >= T) & 
                           (on_ts_org >= filter_ts))
                    
                    if np.any(mask):
                        valid_on_px = on_px[mask]
                        valid_on_qty_org = on_qty_org[mask]  # 使用原始挂单量
                        valid_on_ts_org = on_ts_org[mask]
                        
                        # 计算时间差（毫秒）
                        time_deltas = ts - valid_on_ts_org
                        
                        # 初始化每个时间段的金额总和
                        bucket_amounts = np.zeros(num_buckets, dtype=np.float64)
                        
                        # 分时间段计算金额（只考虑time_buckets内的）
                        for i, t_bound in enumerate(time_buckets):
                            in_bucket = time_deltas < t_bound
                            # 计算该时间桶内的总金额
                            bucket_amounts[i] = np.sum(valid_on_px[in_bucket] * valid_on_qty_org[in_bucket] / 10000)
                            # 将已计算的订单排除（设置为无穷大以便后续桶不再计算）
                            time_deltas[in_bucket] = np.inf
                        
                        # 加权计算（不包含超出time_buckets的部分）
                        weighted_sum = np.sum(bucket_amounts * weights)
                        curr_dataset[index, col] = weighted_sum
                    else:
                        curr_dataset[index, col] = 0  # 无有效挂单记为0
                
                index += 1


# 使用示例：
"""
# 大单金额阈值列表（例如：1000, 5000, 10000 元）
value_thresholds = np.array([1000.0, 5000.0, 10000.0], dtype=np.float64)

# 时间衰减参数列表（例如：0.1, 0.2, 0.3）
decay_list = np.array([0.1, 0.2, 0.3], dtype=np.float64)

# 过滤时间列表（例如：9:25和9:30后的单）
filter_minutes = np.array([25.0, 30.0], dtype=np.float64)

# 结果数组：rows = len(value_thresholds) * len(decay_list) * len(filter_minutes), cols = 2 (bid和ask)
curr_dataset = np.zeros((len(value_thresholds) * len(decay_list) * len(filter_minutes), 2), dtype=np.float64)

# 调用函数
ValueTimeDecayOrgOAModified(
    best_px, on_side, on_px, on_qty_org, on_ts_org, ts,
    value_thresholds, decay_list, filter_minutes, curr_dataset
)
"""