# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:07:37 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import numpy as np
import h5py
import time

def save_one_res(date, res, hf, replace_exist):
    if date in hf and replace_exist:
        del hf[date]
    
    if isinstance(res, np.ndarray):
        # Array storage
        dset = hf.require_dataset(date, shape=res.shape, dtype=res.dtype, track_times=False)
        dset[...] = res
    elif isinstance(res, dict):
        # Dictionary storage
        group = hf.require_group(date)
        for key, value in res.items():
            dset = group.require_dataset(key, shape=value.shape, dtype=value.dtype)
            dset[...] = value

def test_save_speed():
    # Create the test data
    array = np.random.rand(240, 250)
    dict_data = {
        'array1': np.random.rand(240),
        'array2': np.random.rand(240, 250)
    }
    
    # Test saving the array
    with h5py.File('test_array.h5', 'w') as hf:
        start_time = time.time()
        save_one_res('test_array', array, hf, replace_exist=True)
        array_time = time.time() - start_time
    
    # Test saving the dictionary
    with h5py.File('test_dict.h5', 'w') as hf:
        start_time = time.time()
        save_one_res('test_dict', dict_data, hf, replace_exist=True)
        dict_time = time.time() - start_time
    
    # Print the results
    print(f"Saving array took {array_time:.6f} seconds.")
    print(f"Saving dictionary took {dict_time:.6f} seconds.")

if __name__ == "__main__":
    test_save_speed()