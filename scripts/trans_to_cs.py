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


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from core.ft2ts2cs import ConcatProcessor


# %%
# @timeit
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='To initialize Processor.')
    parser.add_argument('-indv', '--ind_ver_name', type=str, help='Indicator Version name')
    parser.add_argument('-nw', '--n_workers', type=int, help='Number of workers', default=1)
    parser.add_argument('-m', '--mode', type=str, help='mode')
    
    args = parser.parse_args()
    ind_ver_name = args.ind_ver_name
    n_workers = args.n_workers
    mode = args.mode

    processor = ConcatProcessor(
        ind_ver_name=ind_ver_name,
        n_workers=n_workers,
        mode=mode,
    )
    processor.run()

if __name__ == "__main__":
    main()