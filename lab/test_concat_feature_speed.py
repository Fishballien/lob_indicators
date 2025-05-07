# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:47:10 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
import numpy as np
import h5py
import os
from pathlib import Path
from functools import partial
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import toml


from utils.dirutils import load_path_config
from utils.market import *
from utils.speedutils import timeit
from utils.timeutils import generate_time_series_in_date_range


# %%
# time: 400s
def concatenate_features_from_parquet1(feature: str, ts_dir: Path, cs_dir: Path):
    feature_dir = ts_dir / feature
    symbols = extract_symbols(feature_dir, '.parquet')

    series_list = []
    
    for symbol in tqdm(symbols, desc=feature):
        series_path = feature_dir / f'{symbol}.parquet'
        series = pd.read_parquet(series_path)
        series.columns = [symbol]
        series_list.append(series)
    
    concatenated_df = pd.concat(series_list, axis=1)
    concatenated_df.to_parquet(cs_dir / f'{feature}.parquet')
    

# load : 1.5min fill: 3min
def concatenate_features_from_parquet2(feature: str, ts_dir: Path, cs_dir: Path, params: dict):
    """
    合并特征数据文件，并基于最大和最小日期动态生成全局索引。
    
    :param feature: 特征名称
    :param ts_dir: 时间序列文件路径
    :param cs_dir: 合并后文件保存路径
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1})
    """
    feature_dir = ts_dir / feature
    symbols = extract_symbols(feature_dir, '.parquet')

    # Step 1: 预加载所有文件到内存并动态更新最大和最小日期
    data_dict = {}
    min_date, max_date = None, None

    for symbol in tqdm(symbols, desc=f"Loading data for {feature}"):
        series_path = feature_dir / f'{symbol}.parquet'
        series = pd.read_parquet(series_path)
        data_dict[symbol] = series

        # 将索引转为日期
        current_min_date = series.index[0].date()
        current_max_date = series.index[-1].date()

        # 更新最小日期和最大日期
        if min_date is None or current_min_date < min_date:
            min_date = current_min_date
        if max_date is None or current_max_date > max_date:
            max_date = current_max_date

    # 确保日期范围有效
    if min_date is None or max_date is None:
        raise ValueError("No valid data found in the input files.")

    # Step 2: 使用生成全局时间序列索引
    global_index = generate_time_series_in_date_range(
        pd.Timestamp(min_date), pd.Timestamp(max_date), params
    )

    # Step 3: 创建一个空的 DataFrame
    concatenated_df = pd.DataFrame(index=global_index, columns=symbols, dtype=float)

    # Step 4: 填充 DataFrame
    for symbol in tqdm(symbols, desc=f"Filling data for {feature}"):
        series = data_dict[symbol].reindex(global_index)
        concatenated_df[symbol] = series  # 假设每个文件只有一列

    # Step 5: 保存结果
    concatenated_df.to_parquet(cs_dir / f'{feature}.parquet')


def read_parquet_file(file_path: Path):
    """读取单个 parquet 文件并返回 DataFrame"""
    series = pd.read_parquet(file_path)
    return series

def reindex_series(series, global_index):
    """对单个 series 进行 reindex"""
    return series.reindex(global_index)

# load : 10s fill: 40s
def concatenate_features_from_parquet3(feature: str, ts_dir: Path, cs_dir: Path, params: dict, max_workers: int = 4):
    """
    合并特征数据文件，并基于最大和最小日期动态生成全局索引。
    
    :param feature: 特征名称
    :param ts_dir: 时间序列文件路径
    :param cs_dir: 合并后文件保存路径
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1})
    :param max_workers: 最大线程数
    """
    feature_dir = ts_dir / feature
    symbols = extract_symbols(feature_dir, '.parquet')

    # Step 1: 使用多线程预加载所有文件到内存并动态更新最大和最小日期
    data_dict = {}
    min_date, max_date = None, None

    file_paths = {symbol: feature_dir / f'{symbol}.parquet' for symbol in symbols}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_parquet_file, path): symbol for symbol, path in file_paths.items()}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading data for {feature}"):
            symbol = futures[future]
            series = future.result()
            data_dict[symbol] = series

            # 将索引转为日期
            current_min_date = series.index[0].date()
            current_max_date = series.index[-1].date()

            # 更新最小日期和最大日期
            if min_date is None or current_min_date < min_date:
                min_date = current_min_date
            if max_date is None or current_max_date > max_date:
                max_date = current_max_date

    # 确保日期范围有效
    if min_date is None or max_date is None:
        raise ValueError("No valid data found in the input files.")

    # Step 2: 使用生成全局时间序列索引
    global_index = generate_time_series_in_date_range(
        pd.Timestamp(min_date), pd.Timestamp(max_date), params
    )

    # Step 3: 创建一个空的 DataFrame
    concatenated_df = pd.DataFrame(index=global_index, columns=symbols, dtype=float)

    # Step 4: 填充 DataFrame
# =============================================================================
#     for symbol in tqdm(symbols, desc=f"Filling data for {feature}"):
#         series = data_dict[symbol].reindex(global_index)
#         concatenated_df[symbol] = series  # 假设每个文件只有一列
#         # series = data_dict[symbol]
#         # concatenated_df.loc[series.index, symbol] = series.iloc[:, 0]  # 假设每个文件只有一列
# =============================================================================
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(reindex_series, data_dict[symbol], global_index): symbol
            for symbol in symbols
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Reindexing data for {feature}"):
            symbol = futures[future]  # 获取当前任务对应的 symbol
            reindexed_series = future.result()
            concatenated_df[symbol] = reindexed_series
            
    # Step 5: 保存结果
    concatenated_df.to_parquet(cs_dir / f'{feature}.parquet')
