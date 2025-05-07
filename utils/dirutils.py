# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:46:31 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import yaml
from pathlib import Path
import shutil


# %% load path
def load_path_config(project_dir):
    path_config_path = project_dir / '.path_config.yaml'
    with path_config_path.open('r') as file:
        path_config = yaml.safe_load(file)
    return path_config


# %%
def list_symbols_in_folder(folder_path):
    folder = Path(folder_path)
    symbols = [file.stem for file in folder.glob("*.parquet")]
    return symbols


# %% clear
def clear_folder(folder_path: Path):
    # 检查文件夹是否存在
    if folder_path.exists() and folder_path.is_dir():
        # 遍历文件夹中的所有文件和子文件夹
        for item in folder_path.iterdir():
            try:
                # 判断是文件还是文件夹
                if item.is_file() or item.is_symlink():
                    item.unlink()  # 删除文件或符号链接
                elif item.is_dir():
                    shutil.rmtree(item)  # 删除文件夹及其所有内容
            except Exception as e:
                print(f'删除 {item} 时出现错误: {e}')
    else:
        print(f'路径 {folder_path} 不存在或不是文件夹')


# %%
def get_filenames_by_extension(folder_path, extension):
    folder = Path(folder_path)
    filenames = [file.stem for file in folder.iterdir() if file.suffix == extension]
    return filenames
