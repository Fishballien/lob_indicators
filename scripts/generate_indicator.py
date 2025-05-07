# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:07:32 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% TODOs
'''
'''
# %% imports
import sys
from pathlib import Path
import argparse
import signal


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from core.processor import *


# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='To initialize Processor.')
    parser.add_argument('-indv', '--ind_ver_name', type=str, help='Indicator Version name')
    parser.add_argument('-p', '--processor', type=str, help='Indicator Processor name')
    parser.add_argument('-sd', '--start_date', type=str, help='start_date')
    parser.add_argument('-ed', '--end_date', type=str, default=None, help='end_date')
    parser.add_argument('-nw', '--n_workers', type=int, help='Number of workers', default=1)
    parser.add_argument('-tn', '--task_n_group', type=int, help='task_n_group', default=1)
    parser.add_argument('-sn', '--save_n_group', type=int, help='save_n_group', default=1)
    parser.add_argument('-rp', '--replace_exist', action='store_true', help='Replace existing data')
    parser.add_argument('-m', '--mode', type=str, help='mode')

    args = parser.parse_args()
    ind_ver_name = args.ind_ver_name
    processor_name = args.processor
    start_date, end_date = args.start_date, args.end_date
    n_workers = args.n_workers
    task_n_group = args.task_n_group
    save_n_group = args.save_n_group
    replace_exist = args.replace_exist
    mode = args.mode

    processor_class = globals()[processor_name]
    processor = processor_class(
        ind_ver_name=ind_ver_name,
        start_date=start_date,
        end_date=end_date,
        n_workers=n_workers,
        task_n_group=task_n_group,
        save_n_group=save_n_group,
        replace_exist=replace_exist,
        mode=mode,
    )
    
    signal.signal(signal.SIGINT, processor.signal_handler)
    signal.signal(signal.SIGTERM, processor.signal_handler)
    
    processor.run()

if __name__ == "__main__":
    main()