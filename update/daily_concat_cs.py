# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:09:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import sys
from datetime import datetime
import yaml
import argparse


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from dirutils import load_path_config
from logutils import FishStyleLogger
from database_handler import DailyUpdateSender, DailyUpdateReader, DailyUpdateMsgSender
from loop_check import CheckDb, ProcessUpdateCoordinator
from dateutils import get_previous_n_trading_day
from update.concat_cs import run_concat


# %%
def daily_concat_cs(config_name=None, delay=1):
    
    # 更新至 ————
    date_today = datetime.today().strftime('%Y%m%d')
    target_date = get_previous_n_trading_day(date_today, delay)
    
    # 读取路径配置
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['workflow_param'])
    
    # 读取参数
    with open(param_dir / 'concat_indicators' / f'{config_name}.yaml', "r") as file:
        params = yaml.safe_load(file)
    # params = toml.load(param_dir / 'update_factors' / f'{update_name}.toml')
    
    # 数据库交互
    # Initialize logger and senders
    mysql_name = params['mysql_name']
    author = params['author']
    log = FishStyleLogger()
    daily_update_sender = DailyUpdateSender(mysql_name, author, log=log)
    daily_update_reader = DailyUpdateReader(mysql_name, log=log)
    msg_sender = DailyUpdateMsgSender(mysql_name, author, log=log)
    
    # Initialize check database and coordinator
    check_db_params = params['check_db_params']
    check_db = CheckDb(daily_update_reader, msg_sender, log, config=check_db_params)
    coordinator = ProcessUpdateCoordinator(check_db, daily_update_sender, msg_sender, log)
    coordinator.set_target_date(target_date)
    
    ## update
        
    # 用更新的股票ind，计算更新的org + org的历史与inc拼接 + 做时序变换
    concat_params = params['concat']
    config_name = concat_params['config_name']
    output = concat_params['output']
    dependency = concat_params['dependency']
    
    with coordinator(output, dependency):
        if not coordinator.skip_task:
            run_concat(config_name)

        
# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--config_name', type=str, help='config_name')
    parser.add_argument('-dl', '--delay', type=int, help='delay')

    args = parser.parse_args()
    config_name, delay = args.config_name, args.delay
    
    
    daily_concat_cs(config_name, delay)

        
# %%
if __name__=='__main__':
    main()