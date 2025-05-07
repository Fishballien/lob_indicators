# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:32:02 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from utils.dirutils import get_filenames_by_extension
from utils.logutils import FishStyleLogger
from utils.datautils import add_dataframe_to_dataframe_reindex, check_dataframe_consistency


# %%
class CheckNUpdate: # TODO: 可以按功能拆分出基类
    
    def __init__(self, historical_dir, incremental_dir, updated_dir, target_names=None, 
                 check_consistency=True, n_workers=1, log=None):
        self.historical_dir = historical_dir
        self.incremental_dir = incremental_dir
        self.updated_dir = updated_dir
        self.target_names = target_names
        self.check_consistency = check_consistency
        self.n_workers = n_workers
        self.log = log or FishStyleLogger()
        
    def run(self):
        self._get_factor_names()
        self._decide_pre_update_dir()
        status = self._update_all_factors()
        return status
        
    def _get_factor_names(self):
        self.factor_names = self.target_names or get_filenames_by_extension(self.historical_dir, '.parquet')
        
    def _decide_pre_update_dir(self):
        self.updated_dir.mkdir(parents=True, exist_ok=True)
        updated_factor_names = get_filenames_by_extension(self.updated_dir, '.parquet')
        
        if all([fac in updated_factor_names for fac in self.factor_names]):
            self.pre_updated_dir = self.updated_dir
        else:
            self.pre_updated_dir = self.historical_dir
            self.log.warning('更新文件数量不足，使用历史文件做更新！')
            
    def _update_all_factors(self):
        """使用多线程并发更新所有因子，遇到错误时立即中止程序"""
        len_of_factors = len(self.factor_names)
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len_of_factors)) as executor:
            futures = {executor.submit(self._update_one_factor, factor_name): 
                       factor_name for factor_name in self.factor_names}
            
            for future in as_completed(futures):
                factor_name = futures[future]
                try:
                    future.result()  # 获取线程的返回结果，确保异常被捕获
                except Exception as exc:
                    self.log.error(f'{factor_name} 更新过程中出错: {exc}')
                    # 记录完整的错误堆栈
                    error_traceback = traceback.format_exc()
                    self.log.error(f'错误堆栈: {error_traceback}')
                    # 立即中止程序
                    raise RuntimeError(f"因子 {factor_name} 更新失败，程序中止") from exc
    
    def _update_one_factor(self, factor_name):
        pre_updated_dir = self.pre_updated_dir
        incremental_dir = self.incremental_dir
        updated_dir = self.updated_dir

        file_name = f'{factor_name}.parquet'
        
        pre_update_path = pre_updated_dir / file_name
        incremental_path = incremental_dir / file_name
        pre_update_data = pd.read_parquet(pre_update_path)
        incremental_data = pd.read_parquet(incremental_path)
        updated_data = self._check_n_update(factor_name, pre_update_data, incremental_data)
        
        updated_path = updated_dir / file_name
        updated_data.to_parquet(updated_path)
    
    def check_n_update(self, factor_name, pre_update_data, incremental_data):
        """
        检查数据一致性并更新数据帧
        
        参数:
        factor_name (str): 因子名称，用于错误信息
        pre_update_data (pd.DataFrame): 待更新的数据帧
        incremental_data (pd.DataFrame): 增量数据帧
        
        返回值:
        pd.DataFrame: 更新后的数据帧
        
        异常:
        ValueError: 当数据一致性检查失败时抛出
        """
        # 检查数据一致性
        if self.check_consistency:
            status, info = check_dataframe_consistency(pre_update_data, incremental_data)
            
            if status == "INCONSISTENT":
                # 构造错误信息
                error_msg = f"因子[{factor_name}]数据一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
                error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                error_msg += f"不一致计数: {info['inconsistent_count']}"
                
                raise ValueError(error_msg)
            
        # 如果一致性检查通过，继续更新数据
        updated_data = self._update_to_updated(pre_update_data, incremental_data)
        
        return updated_data

    def _update_to_updated(self, pre_update_data, incremental_data):
        updated_data = add_dataframe_to_dataframe_reindex(
            pre_update_data, incremental_data)
        
        return updated_data
    
    
def run_concat(config_name):
    """
    Run concatenation process using the specified configuration.
    
    Args:
        config_name (str): Name of the configuration file (without .toml extension)
    """
    path_config = load_path_config(project_dir)
    config_dir = Path(path_config['param']) / 'concat'
    config = toml.load(config_dir / f'{config_name}.toml')
    
    check_consistency = config['check_consistency']
    n_workers = config['n_workers']
    
    for concat_info in config['concat_target']:
        historical_dir = Path(concat_info['historical_dir'])
        incremental_dir = Path(concat_info['incremental_dir'])
        updated_dir = Path(concat_info['updated_dir'])
        target_names = concat_info.get('target_names')
        
        if target_names is not None:
            target_names = [f'{side}_{factor}' for factor in target_names for side in ('Bid', 'Ask')]
        
        c = CheckNUpdate(
            historical_dir, 
            incremental_dir, 
            updated_dir, 
            target_names=target_names,
            check_consistency=check_consistency, 
            n_workers=n_workers
        )
        c.run()
    
    
# %%
if __name__ == "__main__":
    import toml
    from utils.dirutils import load_path_config
    
    config_name = 'concat_for_xkq'

    path_config = load_path_config(project_dir)
    config_dir = Path(path_config['param']) / 'concat'
    config = toml.load(config_dir / f'{config_name}.toml')
    
    check_consistency = config['check_consistency']
    n_workers = config['n_workers']

    for concat_info in config['concat_target']:
        historical_dir = Path(concat_info['historical_dir'])
        incremental_dir = Path(concat_info['incremental_dir'])
        updated_dir = Path(concat_info['updated_dir'])
        target_names = concat_info.get('target_names')
        if target_names is not None:
            target_names = [f'{side}_{factor}' for factor in target_names for side in ('Bid', 'Ask')]
        
        c = CheckNUpdate(historical_dir, incremental_dir, updated_dir, target_names=target_names, 
                         check_consistency=check_consistency, n_workers=n_workers)
        c.run()
        