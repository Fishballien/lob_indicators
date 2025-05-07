# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:23:13 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np


from utils.speedutils import gc_collect_after
from utils.market import index_to_futures


# %% align
def align_columns(main_col, sub_df):
    sub_aligned = pd.DataFrame(columns=main_col, index=sub_df.index)
    inner_columns = main_col.intersection(sub_df.columns)
    sub_aligned[inner_columns] = sub_df[inner_columns]
    return sub_aligned


def align_index(df_1, df_2):
    inner_index = df_1.index.intersection(df_2.index)
    return df_1.loc[inner_index, :], df_2.loc[inner_index, :]


def align_index_with_main(main_index, sub_df):
    sub_aligned = pd.DataFrame(columns=sub_df.columns, index=main_index)
    inner_index = main_index.intersection(sub_df.index)
    sub_aligned.loc[inner_index, :] = sub_df.loc[inner_index, :]
    return sub_aligned


def align_to_primary(df_primary, df_secondary, key_column1, key_column2):
    # 检查主 DataFrame 是否有重复值
    assert not df_primary.duplicated(subset=[key_column1, key_column2]).any(), "df_primary contains duplicate rows based on key_column1 and key_column2"
    
    # 取主 DataFrame 的键值组合
    primary_keys = df_primary[[key_column1, key_column2]]
    
    # 根据主 DataFrame 的键值组合重新索引次要 DataFrame
    df_secondary_aligned = df_secondary.set_index([key_column1, key_column2]).reindex(pd.MultiIndex.from_frame(primary_keys)).reset_index()
    
    return df_secondary_aligned


def align_and_sort_columns(df_list):
    """
    对齐多个 DataFrame 的共同列，并按列名字母顺序重新排列。

    参数:
        df_list (list of pd.DataFrame): 包含多个 DataFrame 的列表。

    返回:
        list of pd.DataFrame: 对齐并重新排列列顺序后的 DataFrame 列表。
    """
    # 找出所有 DataFrame 的共同列
    common_cols = sorted(set.intersection(*(set(df.columns) for df in df_list)))
    
    # 按共同列重新索引每个 DataFrame
    aligned_dfs = [df[common_cols] for df in df_list]
    
    return aligned_dfs


# %% load one
@gc_collect_after
def get_one_factor(process_name=None, factor_name=None, factor_data_dir=None, 
                   price_type='future_twap', normalization_func=None, 
                   date_start=None, date_end=None, 
                   ref_order_col=None, ref_index=None, fix_changed_root=False, debug=0):
    if fix_changed_root:
        str_dir = str(factor_data_dir)
        if str_dir.endswith('neu'):
            factor_data_dir = Path(str_dir + '_4h')
    if debug:
        breakpoint()
    # process_name = 'ma15_sp240'
    factor_dir = factor_data_dir / process_name
    factor_path = factor_dir / f'{factor_name}.parquet'
    try:
        factor = pd.read_parquet(factor_path)
    except:
        print(factor_path)
        traceback.print_exc()
    if price_type in ['future', 'future_twap']:
        factor = factor.rename(columns=index_to_futures)
    factor = normalization_func(factor)
    factor = factor[(factor.index >= date_start) & (factor.index < date_end)]
    if ref_order_col is not None:
        factor = align_columns(ref_order_col, factor)
    if ref_index is not None:
        factor = align_index_with_main(ref_index, factor)
    return factor.fillna(0)


# %% load all & group
def load_all_factors(cluster_info, get_one_factor_func, factor_data_dir, n_workers):
    tasks = []
    
    for index in cluster_info.index:
        process_name, factor_name = cluster_info.loc[index, ['process_name', 'factor']]
        try:
            # 尝试获取特定行和列的值
            factor_dir = Path(cluster_info.loc[index, 'root_dir'])
        except KeyError:
            # 如果列不存在，返回默认值
            factor_dir = factor_data_dir
        tasks.append((index, process_name, factor_name, factor_dir))
        
    factor_dict = {}
    if n_workers is None or n_workers == 1:
        for (index, pool_name, feature_name) in tasks:
            factor_dict[index] = get_one_factor_func(pool_name, feature_name, process_name, factor_name)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(get_one_factor_func, *task[1:-1], factor_data_dir=task[-1]): 
                       task[0] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc='load all factors 📊'):
                idx_to_save = futures[future]
                factor = future.result()
                factor_dict[idx_to_save] = factor
    return factor_dict
                
                
# @gc_collect_after # TODO：之后要改加入root dir
def load_one_group(group_num, group_info, factor_dict={}, normalization_func=None, to_mask=None):
    try:
        # group_name = f'group_{int(group_num)}'
        len_of_group = len(group_info)
        group_factor = None
        for id_, index in enumerate(group_info.index):
            process_name, factor_name, direction = group_info.loc[index, ['process_name', 'factor', 'direction']]
            factor = factor_dict[index]
            # factor = normalization_func(factor).fillna(0)
            if factor is None:
                len_of_group -= 1
                print(process_name, factor_name)
                continue
            factor = factor * direction
            if group_factor is None:
                group_factor = factor
            else:
                group_factor += factor
        group_factor = group_factor / len_of_group
    except:
        traceback.print_exc()
    return int(group_num), group_factor


# %% merge
def add_dataframe_to_dataframe_reindex(df, new_data):
    """
    使用 reindex 将新 DataFrame 的数据添加到目标 DataFrame 中，支持动态扩展列和行，原先没有值的地方填充 NaN。

    参数:
    df (pd.DataFrame): 目标 DataFrame。
    new_data (pd.DataFrame): 要添加的新 DataFrame。

    返回值:
    df (pd.DataFrame): 更新后的 DataFrame。
    """
    # 同时扩展行和列，并确保未填充的空值为 NaN，按排序
    df = df.reindex(index=df.index.union(new_data.index, sort=True),
                    columns=df.columns.union(new_data.columns, sort=True),
                    fill_value=np.nan)
    
    # 使用 loc 添加新数据
    df.loc[new_data.index, new_data.columns] = new_data

    return df


# %% average
# def compute_dataframe_dict_average(df_dict):
#     """
#     计算 df_dict 中每个 DataFrame 对应列的平均值，返回一个包含平均值的 DataFrame。
#     对于每个 DataFrame，先按第一个 DataFrame 的 index 和 columns 进行 reindex，再求平均值。
#     如果某个位置的值为 NaN，则该位置的平均值也为 NaN。
    
#     参数:
#     df_dict (dict): 字典，其中的每个值是一个 DataFrame。
    
#     返回:
#     pd.DataFrame: 包含各列平均值的 DataFrame，index 保持原 DataFrame 的 index。
#     """
#     # 获取第一个 DataFrame 的列名和 index
#     first_df = list(df_dict.values())[0]
#     columns = first_df.columns
#     index = first_df.index

#     # 初始化存储平均值的字典
#     averages = {}

#     # 对每一列进行计算
#     for col in columns:
#         # 对每个 DataFrame 进行 reindex，确保它们的 index 和 columns 与第一个 DataFrame 对齐
#         reindexed_dfs = [df.reindex(index=index, columns=columns) for df in df_dict.values()]

#         # 对于每个位置的列值求和，使用 np.sum 忽略 NaN
#         sum_values = np.sum([df[col].values for df in reindexed_dfs], axis=0)

#         # 检查每个位置是否有 NaN，若有 NaN，则该位置的平均值为 NaN
#         is_nan = np.any([np.isnan(df[col].values) for df in reindexed_dfs], axis=0)

#         # 如果有 NaN，则对应位置的平均值为 NaN
#         averages[col] = np.where(is_nan, np.nan, sum_values / len(df_dict))

#     # 将结果转换为一个 DataFrame，设置 index 为第一个 df 的 index
#     average_df = pd.DataFrame(averages, index=index)
    
#     return average_df


def compute_dataframe_dict_average(df_dict, weight_dict):
    """
    计算 df_dict 中每个 DataFrame 对应列的加权平均值，返回一个包含加权平均值的 DataFrame。
    使用 df_dict 的每个 DataFrame 的 index 和 columns 的交集生成最终的结果，
    并使用给定的 weight_dict 对各 DataFrame 进行加权平均。
    
    参数:
    df_dict (dict): 字典，其中的每个值是一个 DataFrame。
    weight_dict (dict): 字典，其中的每个值是对应 DataFrame 的权重。
    
    返回:
    pd.DataFrame: 包含加权平均值的 DataFrame，index 和 columns 是 df_dict 的交集。
    """
    # 获取所有 DataFrame 的 index 和 columns 的交集
    common_index = df_dict[list(df_dict.keys())[0]].index
    common_columns = df_dict[list(df_dict.keys())[0]].columns

    for df in df_dict.values():
        common_index = common_index.intersection(df.index)
        common_columns = common_columns.intersection(df.columns)

    # 归一化权重，使总和为 1
    total_weight = sum(weight_dict.values())
    normalized_weights = {key: weight_dict[key] / total_weight for key in weight_dict}

    # 初始化存储加权平均值的字典
    weighted_averages = {}

    # 对每一列进行计算
    for col in common_columns:
        weighted_sum = np.zeros(len(common_index))  # 存储加权和
        total_weight_for_column = 0  # 用于追踪权重总和（可以用于调试）

        # 对每个 DataFrame 按照权重加权计算
        for key, df in df_dict.items():
            # 确保该 DataFrame 有这个列，并且索引也在交集中
            if col in df.columns:
                # 先对该 DataFrame 进行 reindex，使其索引与交集一致
                reindexed_df = df.reindex(index=common_index, columns=common_columns)
                
                # 获取该列的加权值，忽略 NaN
                col_values = reindexed_df[col].values
                weight = normalized_weights.get(key, 0)
                weighted_sum += weight * col_values
                
                # 更新权重总和
                total_weight_for_column += weight
        
        # 计算加权平均值（如果有任何 NaN，则加权平均也为 NaN）
        weighted_averages[col] = np.where(np.isnan(weighted_sum), np.nan, weighted_sum)

    # 将结果转换为一个 DataFrame，设置 index 为共同的 index，columns 为共同的 columns
    average_df = pd.DataFrame(weighted_averages, index=common_index)

    return average_df


# =============================================================================
# df1 = pd.DataFrame({
#     'col1': [1, 2, 3],
#     'col2': [4, 5, 6]
# }, index=[10, 11, 12])
# 
# df2 = pd.DataFrame({
#     'col1': [7, np.nan, 9],  # 第二行 col1 为 NaN
#     'col3': [10, 11, 12]
# }, index=[10, 11, 12])
# 
# predict_dict = {('model1', 'test1'): df1, ('model2', 'test2'): df2}
# 
# # 调用函数计算列的平均值
# average_df = compute_dataframe_dict_average(predict_dict)
# 
# print(average_df)
# =============================================================================


# %%
def check_dataframe_consistency(df, new_data):
    """
    使用矩阵运算检查两个DataFrame在重叠的索引部分和合并后的列上是否完全一致。
    完全一致的定义:
    - 两个值都是非NA且相等
    - 两个值都是NA
    - 如果一个值是NA而另一个不是，则视为不一致
    
    参数:
    df (pd.DataFrame): 目标 DataFrame。
    new_data (pd.DataFrame): 要检查的新 DataFrame。
    
    返回值:
    tuple: (status, info)
        - status (str): 'CONSISTENT' 表示数据一致或没有重叠；'INCONSISTENT' 表示存在不一致
        - info (dict): 当status为'INCONSISTENT'时，包含不一致的详细信息；否则为空字典
    """
    # 获取重叠的索引
    overlapping_indices = df.index.intersection(new_data.index)
    
    # 如果没有重叠的索引，直接返回一致状态
    if len(overlapping_indices) == 0:
        return "CONSISTENT", {}
    
    # 获取要检查的列（仅检查new_data中存在的列）
    columns_to_check = df.columns.intersection(new_data.columns)
    
    # 如果没有重叠的列，直接返回一致状态
    if len(columns_to_check) == 0:
        return "CONSISTENT", {}
    
    # 提取重叠部分的数据
    df_overlap = df.loc[overlapping_indices, columns_to_check]
    new_data_overlap = new_data.loc[overlapping_indices, columns_to_check]
    
    # 检查NA的一致性
    df_is_na = df_overlap.isna()
    new_is_na = new_data_overlap.isna()
    
    # NA状态应该一致（都是NA或都不是NA）
    na_inconsistent = (df_is_na != new_is_na)
    
    # 检查非NA值的一致性
    values_inconsistent = (df_overlap != new_data_overlap) & (~df_is_na) & (~new_is_na)
    
    # 合并两种不一致情况
    inconsistent_mask = na_inconsistent | values_inconsistent
    
    # 如果有不一致的元素
    if inconsistent_mask.any().any():
        # 找到第一个不一致的位置
        inconsistent_positions = [(idx, col) for idx, col in zip(
            *np.where(inconsistent_mask.values)
        )]
        
        # 获取第一个不一致的位置和值
        first_pos = inconsistent_positions[0]
        first_idx = overlapping_indices[first_pos[0]]
        first_col = columns_to_check[first_pos[1]]
        
        # 获取不一致的值
        df_value = df.loc[first_idx, first_col]
        new_value = new_data.loc[first_idx, first_col]
        
        # 创建详细信息字典
        info = {
            "index": first_idx,
            "column": first_col,
            "original_value": df_value,
            "new_value": new_value,
            "inconsistent_count": inconsistent_mask.sum().sum()
        }
        
        return "INCONSISTENT", info
    
    # 如果代码执行到这里，说明所有重叠部分都是一致的
    return "CONSISTENT", {}
