# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:30:53 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd


# %%
path = r''
# data = pd.read_csv('your_data.csv', parse_dates=['timestamp'])


# %%
# Step 1: 延迟 30 秒（向后移10个3秒的数据点）
data['mid_price_shifted'] = data['mid_price'].shift(-10)

# Step 2: 将时间列按分钟取整（例如，11:05:23 -> 11:05）
data['minute'] = data['timestamp'].dt.floor('T')

# Step 3: 按分钟分组并计算延迟 30 秒后每分钟的 TWAP
twap_per_minute = (
    data.groupby('minute')['mid_price_shifted']
    .mean()
    .reset_index()
    .rename(columns={'mid_price_shifted': 'twap'})
)

print(twap_per_minute)
