# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:54:53 2024

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
from datetime import datetime
import importlib
import toml


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from core.processor import get_info_fr_params


# %%
data_dir = Path(r'D:\CNIndexFutures\timeseries\lob_indicators\sample_data')

# symbol = '600519.XSHG'
# date = '2024-10-29'

# symbol = '002549.XSHE'
# date = '2024-10-09'

# symbol = '688575.XSHG'
# date = '2021-05-21'

# symbol = '600698.XSHG'
# date = '2020-01-02'

# symbol = '002878.XSHE'
# date = '2020-01-03'

# symbol = '688237.XSHG'
# date = '2024-03-04'

# symbol = '002559.XSHE'
# date = '2023-07-25'

# symbol = '688202.XSHG'
# date = '2020-01-15'

# symbol = '600039.XSHG'
# date = '2020-01-06'

# symbol = '603656.XSHG'
# date = '2020-01-02'

# symbol = '603398.XSHG'
# date = '2020-01-20'

# symbol = '002013.XSHE'
# date = '2020-01-13'

# symbol = '000540.XSHE'
# date = '2020-01-06'

# symbol = '002505.XSHE'
# date = '2020-01-17'

# symbol = '601816.XSHG'
# date = '2020-01-16'

# symbol = '512660.XSHG'
# date = '2020-01-16'

# symbol = '512760.XSHG'
# date = '2020-01-16'

# symbol = "600123.XSHG"
# date = '2023-05-16'

# symbol = "603001.XSHG"
# date = '2021-08-10'

# symbol = "600300.XSHG"
# date = '2022-01-21'

# symbol = "603958.XSHG"
# date = '2024-08-27'

# symbol = "512690.XSHG"
# date = '2024-07-03'

# symbol = "689009.XSHG"
# date = '2024-07-03'

# symbol = "689009.XSHG"
# date = '2024-07-03'

# symbol = "002239.XSHE"
# date = '2024-12-17'

# symbol = "000001.XSHE"
# date = '2024-04-26'

symbol = "600600.XSHG"
date = '2024-04-26'

# symbol = '000016.XSHE'
# date = '2024-04-26'

# symbol = '000301.XSHE'
# date = '2024-04-26'

# symbol = '688800.XSHG'
# date = '2024-04-26'

# symbol = '002245.XSHE'
# date = '2024-04-26'

# symbol = '688137.XSHG'
# date = '2024-04-26'

# symbol = '000023.XSHE'
# date = '2024-04-26'

# symbol = "300036.XSHE"
# date = '2024-11-04'

# ind_cate = 'indicators.total_amount.indv0_total_amount'
# ind_name = 'Indv0'

# ind_cate = 'indicators.prices.prices_v0'
# ind_name = 'PriceV0'

# ind_cate = 'core.auto_generate'
# ind_name = 'GroupGenerate'

ind_cate = 'core.auto_generate_full'
ind_name = 'GGCutPriceRange'

# ind_cate = 'core.auto_generate'
# ind_name = 'GGCutPriceRangeNOrderAmount'

# ind_cate = 'core.auto_generate'
# ind_name = 'GGCutOrderAmount'

# ind_cate = 'core.save_lob'
# ind_name = 'GroupGenerate'

# ind_cate = 'core.trade_only'
# ind_name = 'ProcessTwap'

param = {
    # 'target_ts': {'seconds': 60},
    'target_ts': {'minutes': 1},
    # 'l_thres': 200000,
    }

# batch_name = 'cc_top5_ver0'
# batch_name = 'Batch10_fix_best_241218_selected_f64'
# batch_name = 'test_plot_lob'
# batch_name = 'Batch18_exnoon_250524'
# batch_name = 'Batch18_250425'
batch_name = 'Batch30_250603'
param_dir = Path(r'D:/CNIndexFutures/timeseries/lob_indicators/param')
shared_param_dir = param_dir / 'shared'

param_path = param_dir / f'{batch_name}.toml'
param = toml.load(param_path)
if 'shared_param_name' in param:
    shared_param_name = param['shared_param_name']
    shared_param = toml.load(shared_param_dir / f'{shared_param_name}.toml')
    param['shared_param'] = shared_param
ind_cates, view_infos, indxview_count, factor_list, factor_idx_mapping = get_info_fr_params(param)
param['ind_cates'] = ind_cates
param['view_infos'] = view_infos
param['indxview_count'] = indxview_count
param['factor_idx_mapping'] = factor_idx_mapping


# %%
start = datetime.now() 


# %%
trade_path = data_dir / 'trade' / f'{symbol}.parquet'
order_path = data_dir / 'order' / f'{symbol}.parquet'
trade_data = pd.read_parquet(trade_path)
order_data = pd.read_parquet(order_path)


# %% main
ind_class = getattr(importlib.import_module(ind_cate), ind_name)
start_init = datetime.now() 
go = ind_class(symbol, date, order_data, trade_data, param)
end_init = datetime.now()
print('init', end_init-start_init)
# go = ind_class(date, symbol.name)
start_run = datetime.now() 
res = go.run()
end_run = datetime.now()
print('run', end_run-start_run)
    
    
# %%
end = datetime.now()
print('all', end-start)


# %%
no = 2011000001566048
order_select = order_data[order_data['OrderNo']==no]
trade_select = trade_data[trade_data['buyno']==no]
trade_select = trade_data[trade_data['sellno']==no]

px = 18020
order_select = order_data[(order_data['OrderPx']==px) & (order_data['Side']==b'S')]
trade_select = trade_data[trade_data['tradp']==px]

order_select['Side'] = order_select['Side'].apply(lambda x: x.decode())
order_select['OrderType'] = order_select['OrderType'].apply(lambda x: x.decode())
trade_select['Side'] = trade_select['Side'].apply(lambda x: x.decode())


# %%
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_bid_ask_factors(indicator_array, factor_list):
    """
    绘制bid/ask因子数据的可视化图表
    
    参数:
    indicator_array: numpy数组，每列对应factor_list中的一个因子
    factor_list: 列表，包含因子名称
    """
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义颜色和线型
    # 使用基本颜色列表，兼容老版本matplotlib
    base_colors = ['blue', 'red', 'green', 'orange', 'purple', 
                   'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = base_colors * 2  # 确保有足够的颜色
    bid_linestyle = '-'   # 实线表示Bid
    ask_linestyle = '--'  # 虚线表示Ask
    
    # 提取唯一的参数组合（用于颜色分配）
    param_groups = {}
    color_index = 0
    
    # 处理每个因子
    for i, factor_name in enumerate(factor_list):
        # 提取参数部分（去掉Bid_或Ask_前缀）
        if factor_name.startswith('Bid_'):
            param_part = factor_name[4:]  # 去掉'Bid_'
            line_type = 'Bid'
        elif factor_name.startswith('Ask_'):
            param_part = factor_name[4:]  # 去掉'Ask_'
            line_type = 'Ask'
        else:
            param_part = factor_name
            line_type = 'Unknown'
        
        # 为相同参数组合分配相同颜色
        if param_part not in param_groups:
            param_groups[param_part] = colors[color_index % len(colors)]
            color_index += 1
        
        color = param_groups[param_part]
        linestyle = bid_linestyle if line_type == 'Bid' else ask_linestyle
        
        # 绘制线条
        x_values = range(len(indicator_array))
        y_values = indicator_array[:, i]
        
        # 创建标签
        label = f"{line_type} - {param_part}"
        
        ax.plot(x_values, y_values, 
                color=color, 
                linestyle=linestyle, 
                linewidth=2,
                label=label,
                alpha=0.8)
    
    # 设置图表属性
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Indicator Value', fontsize=12)
    ax.set_title('Bid/Ask Factor Data Visualization', fontsize=14, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    # 设置图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 使用科学计数法显示y轴
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 调整布局
    plt.tight_layout()
    
    return fig, ax

def plot_grouped_bid_ask(indicator_array, factor_list):
    """
    另一种绘图方式：按参数组合分组显示
    """
    # 解析因子名称，按参数组合分组
    bid_ask_pairs = {}
    
    for i, factor_name in enumerate(factor_list):
        if factor_name.startswith('Bid_'):
            param_part = factor_name[4:]
            if param_part not in bid_ask_pairs:
                bid_ask_pairs[param_part] = {'bid_idx': None, 'ask_idx': None}
            bid_ask_pairs[param_part]['bid_idx'] = i
        elif factor_name.startswith('Ask_'):
            param_part = factor_name[4:]
            if param_part not in bid_ask_pairs:
                bid_ask_pairs[param_part] = {'bid_idx': None, 'ask_idx': None}
            bid_ask_pairs[param_part]['ask_idx'] = i
    
    # 创建子图
    n_pairs = len(bid_ask_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4*n_pairs))
    
    if n_pairs == 1:
        axes = [axes]
    
    colors = base_colors * 2  # 确保有足够的颜色
    x_values = range(len(indicator_array))
    
    for idx, (param_name, indices) in enumerate(bid_ask_pairs.items()):
        ax = axes[idx]
        color = colors[idx]
        
        # 绘制Bid数据
        if indices['bid_idx'] is not None:
            bid_data = indicator_array[:, indices['bid_idx']]
            ax.plot(x_values, bid_data, 
                   color=color, linestyle='-', linewidth=2, 
                   label=f'Bid', alpha=0.8)
        
        # 绘制Ask数据
        if indices['ask_idx'] is not None:
            ask_data = indicator_array[:, indices['ask_idx']]
            ax.plot(x_values, ask_data, 
                   color=color, linestyle='--', linewidth=2, 
                   label=f'Ask', alpha=0.8)
        
        ax.set_title(f'{param_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    return fig, axes


fig1, ax1 = plot_bid_ask_factors(res['indicator'], factor_list)
plt.show()