# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:12:06 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import toml
import json
import traceback
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


from utils.logutils import FishStyleLogger
from utils.dirutils import get_filenames_by_extension, load_path_config
from core.processor import *
from core.ft2ts2cs import ConcatProcessor


# %% incremental
class IncrementalUpdate:
    
    dt_format = '%Y-%m-%d'
    
    def __init__(self, incremental_name):
        self.incremental_name = incremental_name
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param'])
        self.flag_dir = Path(self.path_config['flag']) / self.incremental_name
        self.flag_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / 'inc' / f'{self.incremental_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def run(self, date_today):
        date_today = datetime.now().date() if date_today is None else datetime.strptime(date_today, '%Y%m%d')
        self._get_start_date(date_today)
        self._get_end_date(date_today)
        self.log.info(f'Update {self.start_date} ~ {self.end_date}')
        self.msgs = []
        self._load_flags(date_today)
        self._inc_update_all_indicators()
        self._save_flags(date_today)
        self._check_update_status(date_today)
        
    def _get_start_date(self, date_today):
        start_date = self.params.get('start_date')
        if start_date is None:
            lookback = timedelta(**self.params['lookback'])
            start_date = (date_today - lookback).strftime(self.dt_format)
        self.start_date = start_date
        
    def _get_end_date(self, date_today):
        end_date = self.params.get('end_date')
        if end_date is None:
            lookback = timedelta(**self.params['delay'])
            end_date = (date_today - lookback).strftime(self.dt_format)
        self.end_date = end_date
        
    def _load_flags(self, date_today):
        indicators = [ind_info['ind_ver_name'] for ind_info in self.params['indicators']]
        date_in_fmt = date_today.strftime(self.dt_format)
        flag_file_name = f'{date_in_fmt}.json'
        self.flag_path = self.flag_dir / flag_file_name
        if os.path.exists(self.flag_path):
            with open(self.flag_path, 'r') as f:
                self.flags = json.load(f)
        else:
            self.flags = {ind: False for ind in indicators}
        
    def _inc_update_all_indicators(self):
        indicators = self.params['indicators']

        for indicator_info in indicators:
            ind_ver_name = indicator_info['ind_ver_name']
            if self.flags[ind_ver_name]:
                continue
            processor_name = indicator_info['processor_name']
            gen_status = self._generate_one(ind_ver_name, processor_name)
            if gen_status == 0:
                trans_status = self._trans_one(ind_ver_name)
                if trans_status == 0:
                    self.flags[ind_ver_name] = True
                else:
                    self.msgs.append({
                        'type': 'msg',
                        'content': {
                            'level': 'error',
                            'title': f'trans {ind_ver_name} error',
                            'msg': trans_status,
                            }
                        })
            else:
                self.msgs.append({
                    'type': 'msg',
                    'content': {
                        'level': 'error',
                        'title': f'generate {ind_ver_name} error',
                        'msg': gen_status,
                        }
                    })
            
    def _generate_one(self, ind_ver_name, processor_name):
        self.log.info(f'Start generating {ind_ver_name}')
        
        generate_n_workers = self.params['generate_n_workers']
        task_n_group = self.params['task_n_group']
        save_n_group = self.params['save_n_group']
        
        processor_class = globals()[processor_name]
        processor = processor_class(
            ind_ver_name=ind_ver_name,
            start_date=self.start_date,
            end_date=self.end_date,
            n_workers=generate_n_workers,
            task_n_group=task_n_group, 
            save_n_group=save_n_group, 
            mode='update',
        )
        
        try:
            status = processor.run()
            if status == 0:
                self.log.success(f'Finished generating {ind_ver_name}')
            return status
        except:
            e_format = traceback.format_exc()
            self.log.exception(f'Error generating {ind_ver_name}')
            return e_format
       
        
    def _trans_one(self, ind_ver_name):
        self.log.info(f'Start transforming {ind_ver_name}')
        
        trans_n_workers = self.params.get('trans_n_workers')
        if trans_n_workers is None:
            return 0
        
        processor = ConcatProcessor(
            ind_ver_name=ind_ver_name,
            n_workers=trans_n_workers,
            mode='update',
        )
        
        try:
            processor.run()
            self.log.success(f'Finished transforming {ind_ver_name}')
            return 0
        except:
            e_format = traceback.format_exc()
            self.log.exception(f'Error transforming {ind_ver_name}')
            return e_format
        
    def _save_flags(self, date_today):
        with open(self.flag_path, 'w') as f:
            json.dump(self.flags, f)
            
    def _check_update_status(self, date_today):
        if all(self.flags.values()):
            self.msgs.append({
                'type': 'update',
                'content': {
                    'obj': f'ind_inc_{self.incremental_name}',
                    'data_ts': date_today,
                    }
                })
        

# %% update
def format_timedelta_threshold(timedelta_threshold):
    """格式化时间阈值，根据天数或小时输出合适的描述"""
    if timedelta_threshold.days > 0:
        return f"{timedelta_threshold.days}天"
    elif timedelta_threshold.seconds >= 3600:
        hours = timedelta_threshold.seconds // 3600
        return f"{hours}小时"
    elif timedelta_threshold.seconds >= 60:
        minutes = timedelta_threshold.seconds // 60
        return f"{minutes}分钟"
    else:
        return f"{timedelta_threshold.seconds}秒"


def print_daily_diff_stats(difference_mask, combined_index, comparison_df):
    """按天聚合并返回每天不一致行的数量、总行数，以及每行差异比例在一天中的均值，格式为字典"""
    diff_summary = {}
    if difference_mask.any():
        # 将索引转换为日期，并统计每一天不一致的行数
        diff_df = pd.DataFrame({'diff': difference_mask}, index=combined_index)
        diff_df['date'] = diff_df.index.date
        
        daily_diff_count = diff_df.groupby('date')['diff'].sum()  # 每天不一致的数量
        daily_total_count = diff_df.groupby('date')['diff'].count()  # 每天的总数
        
        # 对每一列按日期分组，然后计算每行的比例均值
        row_diff_mean_by_day = comparison_df.groupby(comparison_df.index.date).mean()  # 按日期计算每列的均值
        row_diff_mean = row_diff_mean_by_day.mean(axis=1)  # 对每行计算均值
        
        # 生成差异字典，包含每日不一致比例和每行差异比例的均值
        for date, diff_count in daily_diff_count.items():
            if diff_count > 0:
                total_count = daily_total_count.loc[date]
                daily_mean = row_diff_mean.loc[date]  # 获取当天每行差异比例的均值
                diff_summary[date.strftime('%Y-%m-%d')] = (f"{int(diff_count)}/{total_count}", round(daily_mean, 5))
    
    return diff_summary


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


class CheckNUpdate:
    
    def __init__(self, historical_dir, incremental_dir, updated_dir, params={}, n_workers=1, log=None):
        self.historical_dir = historical_dir
        self.incremental_dir = incremental_dir
        self.updated_dir = updated_dir
        self.params = params
        self.n_workers = n_workers
        self.log = log
        
        self._preprocess_params()
        
    def _preprocess_params(self):
        params = self.params
        timedelta_threshold = params.get('timedelta_threshold', {'minutes': 0})
        params['timedelta_threshold'] = timedelta(**timedelta_threshold)
        params['precision'] = params.get('precision', 1e-5)
        
    def run(self):
        self._get_factor_names()
        self._decide_pre_update_dir()
        self._update_all_factors()
        
    def _get_factor_names(self):
        self.factor_names = get_filenames_by_extension(self.historical_dir, '.parquet')
        
    def _decide_pre_update_dir(self):
        updated_factor_names = get_filenames_by_extension(self.updated_dir, '.parquet')

        len_his = len(self.factor_names)
        len_updated = len(updated_factor_names)
        if len_updated < len_his:
            self.pre_updated_dir = self.historical_dir
            self.log.warning(f'更新文件数量不足 {len_updated}/{len_his}，使用历史文件做更新！')
        else:
            self.pre_updated_dir = self.updated_dir
            
# =============================================================================
#     def _update_all_factors(self):
#         for factor_name in self.factor_names:
#             try:
#                 self._update_one_factor(factor_name)
#             except:
#                 pass
# =============================================================================
            
    def _update_all_factors(self):
        """使用多线程并发更新所有因子"""
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
    
    def _check_n_update(self, factor_name, pre_update_data, incremental_data):
        timedelta_threshold = self.params['timedelta_threshold']
        
        # 检查数据重叠
        pre_start, pre_end, inc_start, inc_end = self._check_data_overlap(
            factor_name, pre_update_data, incremental_data)

        # 处理 before_threshold 数据差异
        self._process_threshold_data(factor_name, pre_update_data, incremental_data, inc_start, 
                                     is_before_threshold=True)
        
        # 处理 after_threshold 数据差异，只到 pre_end
        self._process_threshold_data(factor_name, pre_update_data, incremental_data, inc_start, 
                                     is_before_threshold=False, pre_end=pre_end)
        
        updated_data = self._update_to_updated(pre_update_data, incremental_data, 
                                               inc_start, timedelta_threshold, pre_end)
        
        return updated_data
    
    def _check_data_overlap(self, factor_name, pre_update_data, incremental_data):
        pre_start = pre_update_data.index[0]
        pre_end = pre_update_data.index[-1]
        inc_start = incremental_data.index[0]
        inc_end = incremental_data.index[-1]
        
        if inc_end <= pre_start or inc_start >= pre_end:
            self.log.error(f"{factor_name} 数据没有重叠部分")
            raise
        
        return pre_start, pre_end, inc_start, inc_end

    def _process_threshold_data(self, factor_name, pre_update_data, incremental_data, inc_start, is_before_threshold, 
                               pre_end=None):
        timedelta_threshold = self.params['timedelta_threshold']
        precision = self.params['precision']
        
        threshold_time = inc_start + timedelta_threshold
        timedelta_threshold_in_format = format_timedelta_threshold(timedelta_threshold)
        if is_before_threshold:
            inc_data = incremental_data.loc[inc_start:threshold_time]  # 修正逻辑，from inc_start 开始
            pre_data = pre_update_data.loc[inc_start:threshold_time]
            threshold_desc = f"前{timedelta_threshold_in_format}"
        else:
            inc_data = incremental_data.loc[threshold_time:pre_end]  # 修正逻辑，处理到 pre_end
            pre_data = pre_update_data.loc[threshold_time:pre_end]
            threshold_desc = f"前{timedelta_threshold_in_format}后的"
        
        combined_index = pre_data.index.union(inc_data.index)
        combined_columns = pre_data.columns.union(inc_data.columns)
        
        pre_data = pre_data.reindex(index=combined_index, columns=combined_columns, fill_value=np.nan)
        inc_data = inc_data.reindex(index=combined_index, columns=combined_columns, fill_value=np.nan)
        
        comparison = ~np.isclose(pre_data, inc_data, atol=precision, equal_nan=True)
        difference_rows = comparison.any(axis=1)
        comparison_df = pd.DataFrame(comparison, index=combined_index, columns=combined_columns)
        
        diff_summary = print_daily_diff_stats(difference_rows, combined_index, comparison_df)
        
        if diff_summary:
            self.log.warning(f"{factor_name}{threshold_desc}数据差异: {diff_summary}")
    
    def _update_to_updated(self, pre_update_data, incremental_data, inc_start, timedelta_threshold, pre_end):
        updated_data = add_dataframe_to_dataframe_reindex(
            pre_update_data, incremental_data.loc[(inc_start+timedelta_threshold):])
        
        return updated_data
    
    
# %%
if __name__=='__main__':
    ind_ver_name = "indv9_slope_hl"
    processor_name = 'IndicatorProcessorByL2'
    exchange = "usd"
    
    lob_exchange_dir = Path(r'D:\mnt\Data\Crypto\ProcessedData\lob_shape\usd')
    update_params = {
        'presicion': 1e-5,
        'timedelta_threshold': {
            'days': 0
            },
        }
    n_workers = 2
    
    ind_ver_dir = lob_exchange_dir / ind_ver_name
    historical_dir = ind_ver_dir / 'cs'
    incremental_dir = ind_ver_dir / 'incremental_cs'
    updated_dir = ind_ver_dir / 'updated_cs'
    updated_dir.mkdir(exist_ok=True, parents=True)
    
    log = FishStyleLogger()

    updater = CheckNUpdate(historical_dir, incremental_dir, updated_dir, params=update_params, n_workers=n_workers, log=log)
    updater.run()