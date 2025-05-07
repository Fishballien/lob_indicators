# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:31:32 2024

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
import json


from utils.dirutils import load_path_config
from utils.market import *
from utils.speedutils import timeit
from utils.timeutils import generate_time_series_in_date_range


# %%
def extract_symbols(directory: Path, suffix: str):
    return [f.stem for f in directory.glob(f'*{suffix}')]


# %% ft2ts
def concatenate_features_by_time_series(symbol: str, input_dir: Path, output_dir: Path):
    hdf_file_path = input_dir / f'{symbol}.h5'
    
    with h5py.File(hdf_file_path, 'r') as hf:
        dates = sorted(hf.keys())
        
        arrays = []
        for date in dates:
            data = hf[date][...]
            arrays.append(data)
        
        if len(arrays) == 0:
            print(symbol, 'empty')
            return
        concatenated_array = np.concatenate(arrays, axis=0)
    
    timestamps = concatenated_array['timestamp'].view('datetime64[ms]')
    df = pd.DataFrame(concatenated_array, index=timestamps)
    
    has_duplicates = df.index.duplicated().any()
    if has_duplicates:
        df = df[~df.index.duplicated(keep='first')]
    
    # 将每个feature单独存为Parquet
    for col in df.columns:
        if col != 'timestamp':
            feature_dir = output_dir / col
            feature_dir.mkdir(parents=True, exist_ok=True)
            df[[col]].to_parquet(feature_dir / f'{symbol}.parquet')
            
            
# =============================================================================
# def concatenate_features_by_time_series_with_factor_names(symbol: str, factors, 
#                                                           input_dir: Path, output_dir: Path):
#     hdf_file_path = input_dir / f'{symbol}.h5'
#     
#     with h5py.File(hdf_file_path, 'r') as hf:
#         dates = sorted(hf.keys())
#         
#         ts_arrays = []
#         ind_arrays = []
#         for date in dates:
#             group = hf[date]
#             ts_arrays.append(group['ts'][...])
#             ind_arrays.append(group['indicator'][...])
#         
#         if len(ts_arrays) == 0:
#             print(symbol, 'empty')
#             return
#         ts_array = np.concatenate(ts_arrays, axis=0)
#         ind_array = np.concatenate(ind_arrays, axis=0)
#     
#     timestamps = ts_array.view('datetime64[ms]')
#     df = pd.DataFrame(ind_array, columns=factors, index=timestamps)
#     
#     has_duplicates = df.index.duplicated().any()
#     if has_duplicates:
#         df = df[~df.index.duplicated(keep='first')]
#     
#     # 将每个feature单独存为Parquet
#     for col in df.columns:
#         if col != 'timestamp':
#             feature_dir = output_dir / col
#             feature_dir.mkdir(parents=True, exist_ok=True)
#             df[[col]].to_parquet(feature_dir / f'{symbol}.parquet')
# =============================================================================
            
# =============================================================================
# import time
# 
# def concatenate_features_by_time_series_with_factor_names(symbol: str, factors, 
#                                                           input_dir: Path, output_dir: Path):
#     hdf_file_path = input_dir / f'{symbol}.h5'
#     
#     # 计时开始
#     start_time = time.time()
#     
#     with h5py.File(hdf_file_path, 'r') as hf:
#         dates = sorted(hf.keys())
#         
#         # 计时：读取数据部分
#         read_start_time = time.time()
#         ts_arrays = []
#         ind_arrays = []
#         for date in tqdm(dates, desc="Reading date groups"):
#             group = hf[date]
#             ts_arrays.append(group['ts'][...])
#             ind_arrays.append(group['indicator'][...])
#         read_end_time = time.time()
#         print(f"Time to read data: {read_end_time - read_start_time:.2f} seconds")
#         
#         if len(ts_arrays) == 0:
#             print(symbol, 'empty')
#             return
#         
#         # 计时：拼接时间序列数组
#         concat_start_time = time.time()
#         ts_array = np.concatenate(ts_arrays, axis=0)
#         ind_array = np.concatenate(ind_arrays, axis=0)
#         concat_end_time = time.time()
#         print(f"Time to concatenate arrays: {concat_end_time - concat_start_time:.2f} seconds")
#     
#     # 计时：创建 DataFrame
#     df_start_time = time.time()
#     timestamps = ts_array.view('datetime64[ms]')
#     df = pd.DataFrame(ind_array, columns=factors, index=timestamps)
#     df_end_time = time.time()
#     print(f"Time to create DataFrame: {df_end_time - df_start_time:.2f} seconds")
#     
#     # 计时：去重处理
#     dedup_start_time = time.time()
#     has_duplicates = df.index.duplicated().any()
#     if has_duplicates:
#         df = df[~df.index.duplicated(keep='first')]
#     dedup_end_time = time.time()
#     print(f"Time to remove duplicates: {dedup_end_time - dedup_start_time:.2f} seconds")
#     
# # =============================================================================
# #     # 计时：逐列保存 Parquet 文件
# #     save_start_time = time.time()
# #     for col in tqdm(df.columns, desc="Saving columns to Parquet"):
# #         feature_dir = output_dir / col
# #         feature_dir.mkdir(parents=True, exist_ok=True)
# #         df[[col]].to_parquet(feature_dir / f'{symbol}.parquet')
# #     save_end_time = time.time()
# #     print(f"Time to save columns: {save_end_time - save_start_time:.2f} seconds")
# # =============================================================================
#     
#     # 计时：逐列保存 Parquet 文件
#     save_start_time = time.time()
#     df.to_parquet(output_dir / f'{symbol}.parquet')
#     save_end_time = time.time()
#     print(f"Time to save columns: {save_end_time - save_start_time:.2f} seconds")
#     
#     # 计时结束
#     end_time = time.time()
#     print(f"Total time: {end_time - start_time:.2f} seconds")
# =============================================================================
    
    
def concatenate_features_by_time_series_with_factor_names(symbol: str, factors, 
                                                          input_dir: Path, output_dir: Path):
    hdf_file_path = input_dir / f'{symbol}.h5'
    
    with h5py.File(hdf_file_path, 'r') as hf:
        dates = sorted(hf.keys())
        
        ts_arrays = []
        ind_arrays = []
        for date in dates:
            group = hf[date]
            ts_arrays.append(group['ts'][...])
            ind_arrays.append(group['indicator'][...])
        
        if len(ts_arrays) == 0:
            print(symbol, 'empty')
            return
        ts_array = np.concatenate(ts_arrays, axis=0)
        ind_array = np.concatenate(ind_arrays, axis=0)
    
    timestamps = ts_array.view('datetime64[ms]')
    df = pd.DataFrame(ind_array, columns=factors, index=timestamps)
    
    has_duplicates = df.index.duplicated().any()
    if has_duplicates:
        df = df[~df.index.duplicated(keep='first')]
    
    # 将每个feature单独存为Parquet
    df.to_parquet(output_dir / f'{symbol}.parquet')


# %% ts2cs
# =============================================================================
# def read_parquet_file(file_path: Path):
#     """读取单个 parquet 文件并返回 DataFrame"""
#     series = pd.read_parquet(file_path)
#     return series
# =============================================================================


def reindex_series(series, global_index):
    """对单个 series 进行 reindex"""
    return series.reindex(global_index)


# =============================================================================
# @timeit
# def concatenate_features_from_parquet(feature: str, ts_dir: Path, cs_dir: Path, params: dict, max_workers: int = 4,
#                                       save_executor=None):
#     """
#     合并特征数据文件，并基于最大和最小日期动态生成全局索引。
#     
#     :param feature: 特征名称
#     :param ts_dir: 时间序列文件路径
#     :param cs_dir: 合并后文件保存路径
#     :param params: 时间间隔参数 (字典形式，如 {'seconds': 1})
#     :param max_workers: 最大线程数
#     """
#     feature_dir = ts_dir / feature
#     symbols = extract_symbols(feature_dir, '.parquet')
# 
#     # Step 1: 使用多线程预加载所有文件到内存并动态更新最大和最小日期
#     data_dict = {}
#     min_date, max_date = None, None
# 
#     file_paths = {symbol: feature_dir / f'{symbol}.parquet' for symbol in symbols}
# 
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(read_parquet_file, path): symbol for symbol, path in file_paths.items()}
# 
#         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading data for {feature}"):
#             symbol = futures[future]
#             series = future.result()
#             data_dict[symbol] = series
# 
#             # 将索引转为日期
#             current_min_date = series.index[0].date()
#             current_max_date = series.index[-1].date()
# 
#             # 更新最小日期和最大日期
#             if min_date is None or current_min_date < min_date:
#                 min_date = current_min_date
#             if max_date is None or current_max_date > max_date:
#                 max_date = current_max_date
# 
#     # 确保日期范围有效
#     if min_date is None or max_date is None:
#         raise ValueError("No valid data found in the input files.")
# 
#     # Step 2: 使用生成全局时间序列索引
#     global_index = generate_time_series_in_date_range(
#         pd.Timestamp(min_date), pd.Timestamp(max_date), params
#     )
# 
#     # Step 3: 创建一个空的 DataFrame
#     concatenated_df = pd.DataFrame(index=global_index, columns=symbols, dtype=float)
# 
#     # Step 4: 填充 DataFrame
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(reindex_series, data_dict[symbol], global_index): symbol
#             for symbol in symbols
#         }
# 
#         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Reindexing data for {feature}"):
#             symbol = futures[future]  # 获取当前任务对应的 symbol
#             reindexed_series = future.result()
#             concatenated_df[symbol] = reindexed_series
#             
#     # Step 5: 保存结果
#     # concatenated_df.to_parquet(cs_dir / f'{feature}.parquet')
#     save_path = cs_dir / f'{feature}.parquet'
#     return save_data_with_executor(save_executor, concatenated_df, save_path)
# =============================================================================

def read_parquet_file(file_path: Path, feature):
    """读取单个 parquet 文件并返回 DataFrame"""
    series = pd.read_parquet(file_path, columns=[feature])
    return series


@timeit
def concatenate_features_from_parquet(feature: str, ts_dir: Path, cs_dir: Path, params: dict, max_workers: int = 4,
                                      save_executor=None):
    """
    合并特征数据文件，并基于最大和最小日期动态生成全局索引。
    
    :param feature: 特征名称
    :param ts_dir: 时间序列文件路径
    :param cs_dir: 合并后文件保存路径
    :param params: 时间间隔参数 (字典形式，如 {'seconds': 1})
    :param max_workers: 最大线程数
    """
    symbols = extract_symbols(ts_dir, '.parquet')

    # Step 1: 使用多线程预加载所有文件到内存并动态更新最大和最小日期
    data_dict = {}
    min_date, max_date = None, None

    file_paths = {symbol: ts_dir / f'{symbol}.parquet' for symbol in symbols}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_parquet_file, path, feature): symbol for symbol, path in file_paths.items()}

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
    # concatenated_df.to_parquet(cs_dir / f'{feature}.parquet')
    save_path = cs_dir / f'{feature}.parquet'
    return save_data_with_executor(save_executor, concatenated_df, save_path)


def save_dataframe_to_parquet(df, path):
    """
    保存 DataFrame 到 Parquet 文件。
    """
    df.to_parquet(path)


def save_data_with_executor(executor, df, save_path):
    """
    使用 ThreadPoolExecutor 异步保存 DataFrame。
    :param executor: ThreadPoolExecutor 实例
    :param df: 待保存的 DataFrame
    :param save_path: 保存路径
    """
    return executor.submit(save_dataframe_to_parquet, df, save_path)


# %%
class ConcatProcessor:
    
    mode_folder_mapping = {
        'by_symbol_by_date': {
            'init': 'by_symbol_by_date',
            'update': 'incremental_by_symbol_by_date'
            },
        'by_feature_by_symbol': {
            'init': 'by_feature_by_symbol',
            'update': 'incremental_by_feature_by_symbol'
            },
        'cs': {
            'init': 'cs',
            'update': 'incremental_cs'
            },
        }
    
    def __init__(self, ind_ver_name, n_workers, mode='init'):
        self.ind_ver_name = ind_ver_name
        self.n_workers = n_workers
        self.mode = mode

        # Initialize directories and load indicators
        self._initialize_directories()
        self._load_params()
        
    def _initialize_directories(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        path_config = load_path_config(project_dir)
        self.lob_shape_dir = Path(path_config['lob_indicators'])
        self.ind_ver_dir = self.lob_shape_dir / self.ind_ver_name
        self.raw_feature_dir = self.ind_ver_dir / self.mode_folder_mapping['by_symbol_by_date'][self.mode]
        self.ts_dir = self.ind_ver_dir / self.mode_folder_mapping['by_feature_by_symbol'][self.mode]
        self.cs_dir = self.ind_ver_dir / self.mode_folder_mapping['cs'][self.mode]
        self.ts_dir.mkdir(exist_ok=True, parents=True)
        self.cs_dir.mkdir(exist_ok=True, parents=True)
        self.param_dir = Path(path_config['param'])
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.ind_ver_name}.toml')
        
    def _decide_saved_type(self):
        try:
            with open(self.ind_ver_dir / 'factors.json') as f:
                factors = json.load(f)
            self.concatenate_func = partial(concatenate_features_by_time_series_with_factor_names,
                                            factors = factors, input_dir=self.raw_feature_dir,
                                            output_dir=self.ts_dir)
        except:
            self.concatenate_func = partial(concatenate_features_by_time_series,
                                            input_dir=self.raw_feature_dir,
                                            output_dir=self.ts_dir)

    def trans_to_by_feature_by_symbol(self):
        self._decide_saved_type()
        symbols = extract_symbols(self.raw_feature_dir, '.h5')
        if self.n_workers is None or self.n_workers == 1:
            for symbol in tqdm(symbols, desc='ft2ts'):
                self.concatenate_func(symbol)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self.concatenate_func, symbol) for symbol in symbols]
            for future in tqdm(as_completed(futures), total=len(futures), desc='ft2ts'):
                try:
                    future.result()
                except:
                    traceback.print_exc()

    def trans_to_cs(self):
        try:
            with open(self.ind_ver_dir / 'factors.json') as f:
                features = json.load(f)
        except:
            features = [f for f in self.ts_dir.iterdir() if f.is_dir()]
        
        with ThreadPoolExecutor(max_workers=4) as save_executor:
            futures = []
            for feature in features:
                # future = concatenate_features_from_parquet(feature.name, self.ts_dir, self.cs_dir, self.params['target_ts'],
                #                                            max_workers=self.n_workers, save_executor=save_executor)
                future = concatenate_features_from_parquet(feature, self.ts_dir, self.cs_dir, self.params['target_ts'],
                                                           max_workers=self.n_workers, save_executor=save_executor)
                futures.append(future)
                
            # 等待所有任务完成
            for future in tqdm(as_completed(futures), desc='saving'):
                future.result()  # 捕获和处理异常（如果有）
    
    def run(self):
        self.trans_to_by_feature_by_symbol()
        self.trans_to_cs()
