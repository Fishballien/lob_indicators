# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:01:46 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import os

def rename_parquet_files(directory):
    """
    Renames all parquet files in the given directory, replacing '__' with '_'.
    
    :param directory: Path to the directory containing parquet files.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".parquet") and "__" in filename:
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace("__", "_")
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# Example usage:
directory_path = "/mnt/data1/xintang/index_factors/Batch3_241126_1"  # Replace with the actual path
rename_parquet_files(directory_path)
