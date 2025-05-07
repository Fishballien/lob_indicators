# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:42:02 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
import sys
import numpy as np
from pathlib import Path
import importlib
from functools import partial
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import h5py
import toml
import random
import signal
import json
import copy


# %% import self_defined
from utils.load_data import load_data
from utils.dirutils import load_path_config, list_symbols_in_folder, clear_folder
from utils.timeutils import generate_date_range
from utils.naming import generate_factor_names
from utils.param import para_allocation
from utils.speedutils import timeit


# %% new test
class IndicatorProcessor:
    
    mode_folder_mapping = {
        'init': 
            {
            'by_symbol_by_date': 'by_symbol_by_date',
            'by_feature_by_symbol': 'by_feature_by_symbol',
            'cs': 'cs',
            },
        'update': {
            'by_symbol_by_date': 'incremental_by_symbol_by_date',
            'by_feature_by_symbol': 'incremental_by_feature_by_symbol',
            'cs': 'incremental_cs',
            }
        }
    
    def __init__(self, ind_ver_name, 
                 start_date, end_date,
                 n_workers, replace_exist=True, mode='init'):
        self.ind_ver_name = ind_ver_name
        self.start_date = start_date
        self.end_date = end_date
        self.n_workers = n_workers
        self.replace_exist = replace_exist
        self.mode = mode
        
        self._init_signal_handler()

        # Initialize directories and load indicators
        self._initialize_directories()
        self._load_params()
        self._load_indicator_info()
        self.hf_dict = {}
        self.symbol_path = {}
        
    def _init_signal_handler(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def _initialize_directories(self):
        file_path = Path(__file__).resolve()
        self.project_dir = file_path.parents[1]
        path_config = load_path_config(self.project_dir)
        self.trade_dir = Path(path_config['trade'])
        self.order_dir = Path(path_config['order'])
        self.lob_shape_dir = Path(path_config['lob_indicators'])
        self.param_dir = Path(path_config['param'])
        self.shared_param_dir = self.param_dir / 'shared'
        folder_to_save = self.mode_folder_mapping[self.mode]
        self.ind_dir = self.lob_shape_dir / self.ind_ver_name
        self.save_dir = self.ind_dir / folder_to_save['by_symbol_by_date']
        self.save_dir.mkdir(exist_ok=True, parents=True)
        if self.mode == 'update':
            for folder_type in self.mode_folder_mapping['update']:
                folder_name = self.mode_folder_mapping['update'][folder_type]
                clear_folder(self.ind_dir / folder_name)
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.ind_ver_name}.toml')
        
    def _load_indicator_info(self):
        ind_cate = self.params['ind_cate']
        ind_name = self.params['ind_name']
        
        self.ind_module = importlib.import_module(ind_cate)
        self.ind_class = getattr(self.ind_module, ind_name)
        
# =============================================================================
#     def _preload_funcs_if_needed(self):
#         if 'func_file_path' in self.param:
#             func_file_path = self.param['func_file_path']
#             func_module = importlib.import_module(func_file_path)
#             for ind_cate in self.param:
#                 if isinstance(self.param[ind_cate], dict) and ind_cate != 'target_ts':
#                     globals()[ind_cate] = getattr(func_module, ind_cate)
# =============================================================================

    def _init_process_one_symbol_oneday_info(self, symbol, param):
        pass
    
    def process_one_day(self, date):
        generate_one_symbol_one_day_func, symbols = self._init_process_one_symbol_oneday_info(date, self.params)
        futures = []
        for symbol in symbols:
            if symbol not in self.hf_dict:
                hdf_file_path = self.save_dir / f'{symbol}.h5'
                try:
                    self.hf_dict[symbol] = h5py.File(hdf_file_path, 'a')
                except Exception as e:
                    print(f'{symbol} failed to open h5: {e}')
                    self.hf_dict[symbol] = h5py.File(hdf_file_path, 'w')
            hf = self.hf_dict[symbol]
            if not self.replace_exist:
                if date in list(hf.keys()):
                    continue
            save_func = partial(save_one_res, hf=hf, replace_exist=self.replace_exist)
            futures.append((generate_one_symbol_one_day_func, date, save_func, symbol))
        return futures

    def run(self):
        self.dates = generate_date_range(self.start_date, self.end_date)
        
        try:
            all_futures = []
            for date in tqdm(self.dates, desc='Setting up tasks by dates'):
                date_futures = self.process_one_day(date)
                all_futures.extend(date_futures)
    
            # Shuffle all futures to distribute the load more evenly
            random.shuffle(all_futures)
            
            if self.n_workers is None or self.n_workers == 1: 
                for func, date, save_func, symbol_name in tqdm(all_futures, desc='Single Processing'):
                # for func, date, save_func, symbol_name in all_futures:
                    # print('processing', symbol_name, date)
                    res = func(symbol_name)
                    if res is not None:
                        save_func(date, res)
            else:
                future_to_task = {}
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    for func, date, save_func, symbol_name in all_futures:
                        future = executor.submit(func, symbol_name)
                        future_to_task[future] = (func, date, save_func, symbol_name)
        
                    for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc='Processing'):
                        func, date, save_func, symbol_name = future_to_task[future]
                        try:
                            res = future.result()
                            # print(res)
                            if res is not None:
                                save_func(date, res)
                        except Exception as e:
                            print(f"Error processing {symbol_name} on {date}: {e}")
            return 0
        except:
            traceback.print_exc()
            return 1
        finally:
            self._close_file()

    def _close_file(self):
        # Close all HDF5 files
        for hf in self.hf_dict.values():
            try:
                hf.close()
            except Exception as e:
                print(f"Error closing HDF5 file: {e}")
                
    def signal_handler(self, sig, frame):
        print("收到终止信号，正在清理资源...")
        self._close_file()  # 清理操作，例如关闭文件
        os._exit(0)
        

# @timeit
def save_one_res(date, res, hf, replace_exist):
    if date in hf and replace_exist:
        del hf[date]
    
    if isinstance(res, np.ndarray):
        # 数组存储方式
        dset = hf.require_dataset(date, shape=res.shape, dtype=res.dtype, track_times=False)
        dset[...] = res
    elif isinstance(res, dict):
        # 字典存储方式
        group = hf.require_group(date)
        for key, value in res.items():
            dset = group.require_dataset(key, shape=value.shape, dtype=value.dtype)
            dset[...] = value
            

# %% l2
class IndicatorProcessorByL2(IndicatorProcessor):
    
    def _init_process_one_symbol_oneday_info(self, date, params):
        trade_dir = self.trade_dir / date
        order_dir = self.order_dir / date
        load_trade_func = partial(load_data, data_dir=trade_dir)
        load_order_func = partial(load_data, data_dir=order_dir)
        generate_one_symbol_one_day_func = partial(generate_one_symbol_one_day_l2, date=date,
                                                   load_trade_func=load_trade_func,
                                                   load_order_func=load_order_func,
                                                   params=params, ind_class=self.ind_class,
                                                   )
        trade_symbols = list_symbols_in_folder(trade_dir)
        order_symbols = list_symbols_in_folder(order_dir)
        symbols = list(set(trade_symbols) & set(order_symbols))
        return generate_one_symbol_one_day_func, symbols
    
    
def generate_one_symbol_one_day_l2(symbol, date, load_trade_func, load_order_func, params, ind_class):
    try:
        trade_data = load_trade_func(symbol)
        order_data = load_order_func(symbol)
    except:
        traceback.print_exc()
        print('data', symbol, date)
        sys.stdout.flush()
        return None
    try:
        go = ind_class(symbol, date, order_data, trade_data, params)
        res = go.run()
        return res
    except:
        traceback.print_exc()
        print('process', symbol, date)
        sys.stdout.flush()
        return None
    
    
# %%
class IndicatorProcessorByL2Batch(IndicatorProcessorByL2):
    
    def __init__(self, ind_ver_name, 
                 start_date, end_date,
                 n_workers, replace_exist=True, mode='init'):
        super().__init__(ind_ver_name, 
                         start_date, end_date,
                         n_workers, replace_exist=replace_exist, mode=mode)
        self._preprocess_params()

    def _preprocess_params(self):
        if 'shared_param_name' in self.params:
            shared_param_name = self.params['shared_param_name']
            shared_param = toml.load(self.shared_param_dir / f'{shared_param_name}.toml')
            self.params['shared_param'] = shared_param
        ind_cates, view_infos, indxview_count, factor_list, factor_idx_mapping = get_info_fr_params(self.params)
        self.params['ind_cates'] = ind_cates
        self.params['view_infos'] = view_infos
        self.params['indxview_count'] = indxview_count
        self.params['factor_idx_mapping'] = factor_idx_mapping
        
        with open(self.ind_dir / 'factors.json', 'w') as f:
            json.dump(factor_list, f, indent=4)


def get_info_fr_params(params):
    
    shared_param = params.get('shared_param', {})
    
    ## get_ind_cate_list
    ind_cates = []
    for ind_cate in params:
        if isinstance(params[ind_cate], dict) and ind_cate not in ['target_ts', 'shared_param']:
            ind_cates.append(ind_cate)
                
    ## get view_name_list
    views = []
    for ind_cate in ind_cates:
        if not shared_param:
            views.append(ind_cate)
        else:
            views.extend(generate_factor_names(ind_cate, shared_param))
                
    ## get view_info
    view_names = generate_factor_names('', shared_param)
    view_info_list = para_allocation(shared_param)
    view_infos = dict(zip(view_names, view_info_list))
            
    ## get_ind_names
    factor_name_mapping = {}
    for ind_cate in ind_cates:
        param_dict = params[ind_cate].get('param', {})
        for view_name, view_info in view_infos.items():
            factor_name_mapping[(ind_cate, view_name)] = generate_factor_names(
                f'{ind_cate}_{view_name}', param_dict)
            
    ## get indxview count
    indxview_count = {}
    for k, factor_names in factor_name_mapping.items():
        indxview_count[k] = len(factor_names)
        
    ## get factor_list & mapping
    factor_list = []
    factor_idx_mapping = {}
    idx = 0
    for (ind_cate, view_name), factor_names in factor_name_mapping.items():
        for i_f, factor_name in enumerate(factor_names):
            factor_list.append(f'Bid_{factor_name}')
            factor_idx_mapping[(ind_cate, view_name, i_f, 0)] = idx
            idx += 1
            factor_list.append(f'Ask_{factor_name}')
            factor_idx_mapping[(ind_cate, view_name, i_f, 1)] = idx
            idx += 1

    return ind_cates, view_infos, indxview_count, factor_list, factor_idx_mapping