# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:01:08 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import h5py

path = 'D:/CNIndexFutures/timeseries/lob_indicators/sample_data/fix/002644.XSHE.h5'

# =============================================================================
# try:
#     with h5py.File(path, 'r', swmr=True) as f:
#         print("Successfully opened the file in SWMR mode.")
# except Exception as e:
#     print(f"Failed to open the file in SWMR mode. Error: {e}")
# =============================================================================


# =============================================================================
# try:
#     with h5py.File(path, 'r') as f:
#         print("Successfully opened the file. Attempting to list contents...")
#         def print_structure(name):
#             print(f"Found dataset or group: {name}")
#         f.visit(print_structure)
# except Exception as e:
#     print(f"Failed to open the file. Error: {e}")
# =============================================================================
    
    
# =============================================================================
# import tables
# 
# try:
#     with tables.open_file(path, mode="r") as f:
#         print("Opened file successfully.")
#         print("Contents:")
#         f.walk_nodes("/", classname="Group", func=lambda node: print(node._v_pathname))
# except Exception as e:
#     print(f"Failed to open the file. Error: {e}")
# =============================================================================

try:
    with h5py.File(path, 'r', libver='latest') as f:
        print("Attempting to read partial data...")
        for key in f.keys():
            try:
                print(f"Dataset '{key}': {f[key][:]}")
            except Exception as e:
                print(f"Failed to read dataset '{key}': {e}")
except Exception as e:
    print(f"Failed to open the file. Error: {e}")