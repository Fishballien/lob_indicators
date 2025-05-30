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

symbol = "002239.XSHE"
date = '2024-12-17'

# symbol = "000001.XSHE"
# date = '2024-04-26'

# symbol = "600600.XSHG"
# date = '2024-04-26'

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
batch_name = 'Batch26_250525'
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

