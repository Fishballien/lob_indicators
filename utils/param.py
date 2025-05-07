# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:53:10 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import itertools


# %%
def para_allocation(para_dict):
    # 参数排列组合
    allocated_para_detail = list(itertools.product(*para_dict.values()))
    # 参数组匹配参数名
    allocated_para_detail_with_name = list(
        map(lambda x: dict(zip(para_dict.keys(), list(x))), allocated_para_detail))
    return allocated_para_detail_with_name